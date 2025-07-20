# ui_app.py - Complete Integrated 3D CAD Classification System
import os
import sys
import json
import argparse
import tempfile
import datetime
import torch
import torch.nn.functional as F
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, QUrl, QPropertyAnimation, QEasingCurve
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout,
    QLabel, QFileDialog, QMessageBox, QGroupBox, QFrame, QSizePolicy,
    QGridLayout, QListWidget, QSlider, QCheckBox, QSpacerItem, QColorDialog,
    QDialog
)
from PyQt5.QtGui import QFont, QPalette, QColor, QDragEnterEvent, QDropEvent, QKeyEvent, QImage, QPixmap, QPainter

from OCC.Display.backend import load_backend
from OCC.Core.gp import gp_Dir, gp_Pnt
from OCC.Core.TopoDS import TopoDS_Vertex
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.AIS import AIS_Shaded, AIS_WireFrame, AIS_Shape
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.Prs3d import Prs3d_LineAspect
from OCC.Core.Graphic3d import Graphic3d_NOM_NEON_GNC

load_backend("pyqt5")
from OCC.Display.qtDisplay import qtViewer3d
from OCC.Extend.DataExchange import read_step_file

# Import solid_to_graph functionality directly
import dgl
import numpy as np
from occwl.graph import face_adjacency
from occwl.io import load_step
from occwl.uvgrid import ugrid, uvgrid
from tqdm import tqdm
import signal


def build_graph(solid, curv_num_u_samples=10, surf_num_u_samples=10, surf_num_v_samples=10):
    """Convert a CAD solid to a DGL graph with UV-grid features"""
    # Build face adjacency graph with B-rep entities as node and edge features
    graph = face_adjacency(solid)

    # Compute the UV-grids for faces
    graph_face_feat = []
    for face_idx in graph.nodes:
        # Get the B-rep face
        face = graph.nodes[face_idx]["face"]
        # Compute UV-grids
        points = uvgrid(
            face, method="point", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        normals = uvgrid(
            face, method="normal", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        visibility_status = uvgrid(
            face, method="visibility_status", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary
        # Concatenate channel-wise to form face feature tensor
        face_feat = np.concatenate((points, normals, mask), axis=-1)
        graph_face_feat.append(face_feat)
    graph_face_feat = np.asarray(graph_face_feat)

    # Compute the U-grids for edges
    graph_edge_feat = []
    for edge_idx in graph.edges:
        # Get the B-rep edge
        edge = graph.edges[edge_idx]["edge"]
        # Ignore degenerate edges, e.g. at apex of cone
        if not edge.has_curve():
            continue
        # Compute U-grids
        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)
        # Concatenate channel-wise to form edge feature tensor
        edge_feat = np.concatenate((points, tangents), axis=-1)
        graph_edge_feat.append(edge_feat)
    graph_edge_feat = np.asarray(graph_edge_feat)

    # Convert face-adj graph to DGL format
    edges = list(graph.edges)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
    dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat)
    dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat)
    return dgl_graph


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.current_lang = 'zh'
        self.lang_map = {
            'zh': {
                'title': "3D CAD分类系统",
                'load_model': "加载模型",
                'load_labels': "加载标签",
                'load_step': "加载STEP",
                'clear': "清空显示",
                'help': "帮助",
                'view_categories': "所有类别",
                'history': "历史记录",
                'export_image': "导出图片",
                'control_panel': "控制面板",
                'display_control': "显示控制",
                'display_mode': "显示模式",
                'shaded': "着色",
                'wireframe': "线框",
                'transparency': "透明度",
                'color': "颜色",
                'choose_color': "选择颜色",
                'view_control': "视图控制",
                'iso_view': "等轴视图\n(I)",
                'reset_view': "重置视图\n(ESC)",
                'drop_label': "拖拽STEP文件到此处",
                'copyright': "© 3D CAD分类系统",
                'no_model': "没有可导出的3D模型",
                'preview_title': "图片导出预览",
                'zoom_control': "缩放控制",
                'zoom': "缩放:",
                'save_image': "保存图片",
                'cancel': "取消",
                'history_title': "历史记录",
                'no_history': "没有历史记录",
                'clear_history': "清除所有历史记录",
                'model_loaded': "模型已加载: {}",
                'labels_loaded': "标签已加载: {}",
                'processing': "正在处理...",
                'converting': "正在转换STEP文件...",
                'classifying': "正在进行分类...",
                'no_model_error': "请先加载模型文件",
                'no_labels_error': "请先加载标签映射文件",
                'unsupported_file': "不支持的文件类型",
                'load_error': "加载出错: {}",
                'process_error': "处理出错: {}",
                'convert_error': "转换出错: {}",
                'result_text': "<span style='font-size:14pt; font-weight:bold;'>分类结果: </span><span style='color:#2c3e50; font-size:14pt;'>{}</span><span style='font-size:14pt; font-weight:bold;'> | 置信度: </span><span style='color:#4a6fa5; font-size:14pt;'>{:.1f}%</span>",
                'image_saved': "图片已保存: {}",
                'history_cleared': "已清除所有历史记录",
                'categories_title': "类别列表",
                'no_labels_error': "请先加载标签文件",
                'categories_text': "<b>类别列表:</b><br>{}",
                'error': "错误"
            },
            'en': {
                'title': "3D CAD Classification System",
                'load_model': "Load Model",
                'load_labels': "Load Labels",
                'load_step': "Load STEP",
                'clear': "Clear Display",
                'help': "Help",
                'view_categories': "View Categories",
                'history': "History",
                'export_image': "Export Image",
                'control_panel': "Control Panel",
                'display_control': "Display Control",
                'display_mode': "Display Mode",
                'shaded': "Shaded",
                'wireframe': "Wireframe",
                'transparency': "Transparency",
                'color': "Color",
                'choose_color': "Choose Color",
                'view_control': "View Control",
                'iso_view': "Isometric View\n(I)",
                'reset_view': "Reset View\n(ESC)",
                'drop_label': "Drop STEP File Here",
                'copyright': "© 3D CAD Classification System",
                'no_model': "No 3D model to export",
                'preview_title': "Image Export Preview",
                'zoom_control': "Zoom Control",
                'zoom': "Zoom:",
                'save_image': "Save Image",
                'cancel': "Cancel",
                'history_title': "History",
                'no_history': "No history",
                'clear_history': "Clear All History",
                'model_loaded': "Model loaded: {}",
                'labels_loaded': "Labels loaded: {}",
                'processing': "Processing...",
                'converting': "Converting STEP file...",
                'classifying': "Classifying...",
                'no_model_error': "Please load model file first",
                'no_labels_error': "Please load label mapping file first",
                'unsupported_file': "Unsupported file type",
                'load_error': "Load error: {}",
                'process_error': "Process error: {}",
                'convert_error': "Convert error: {}",
                'result_text': "<span style='font-size:14pt; font-weight:bold;'>Result: </span><span style='color:#2c3e50; font-size:14pt;'>{}</span><span style='font-size:14pt; font-weight:bold;'> | Confidence: </span><span style='color:#4a6fa5; font-size:14pt;'>{:.1f}%</span>",
                'image_saved': "Image saved: {}",
                'history_cleared': "All history cleared",
                'categories_title': "Categories",
                'no_labels_error': "Please load label file first",
                'categories_text': "<b>Categories:</b><br>{}",
                'error': "Error"
            }
        }

        self.title = self.lang_map[self.current_lang]['title']
        self.setWindowTitle(self.title)
        self.resize(1100, 700)
        self.current_model = None
        self.label_mapping = None
        self.ais_list = []
        self.model_loaded = False
        self.labels_loaded = False
        self.step_loaded = False
        self.history = []
        self.history_window = None
        self.last_mouse_pos = None
        self.current_shape = None
        self.default_color = Quantity_Color(0.7, 0.7, 0.7, Quantity_TOC_RGB)

        # Create temp directory for bin files
        self.temp_dir = tempfile.mkdtemp(prefix="cad_classifier_")

        self.setup_ui()
        self.setup_style()

        self.canvas.InitDriver()
        self.display = self.canvas._display
        self.setAcceptDrops(True)
        QTimer.singleShot(300, self.force_refresh_display)

    def setup_style(self):
        self.setStyleSheet("""
            QWidget {
                font-family: 'Microsoft YaHei';
                font-size: 11px;
                background-color: #f5f7fa;
            }
            QPushButton {
                min-width: 70px;
                min-height: 24px;
                padding: 4px 8px;
                border-radius: 4px;
                background-color: #4a6fa5;
                color: white;
                border: none;
            }
            QPushButton:hover { background-color: #5a7fb5; }
            QPushButton:pressed { background-color: #3a5f95; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
            QPushButton[loaded="true"] { background-color: #4caf50; }
            QGroupBox {
                border: 1px solid #d1d9e6;
                border-radius: 5px;
                margin-top: 8px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: #4a6fa5;
                font-weight: bold;
            }
            QFrame#display_frame {
                border: 1px solid #d1d9e6;
                border-radius: 5px;
                background-color: white;
            }
            QLabel#status_label {
                background-color: white;
                color: #2c3e50;
                border: 1px solid #d1d9e6;
                border-radius: 4px;
                padding: 8px;
                min-height: 32px;
                min-width: 200px;
                font-size: 13px;
                font-weight: bold;
                font-family: 'Segoe UI', 'Microsoft YaHei';
            }
            QLabel#status_label[error="true"] {
                background-color: #ffebee;
                color: #c62828;
                border: 1px solid #ef9a9a;
            }
            QLabel#drop_label {
                background-color: rgba(255, 255, 255, 0.9);
                color: #666;
                border: 2px dashed #4a6fa5;
                border-radius: 6px;
                padding: 12px;
                font-size: 12px;
                qproperty-alignment: AlignCenter;
            }
            QListWidget {
                border: 1px solid #d1d9e6;
                border-radius: 4px;
                padding: 3px;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:hover {
                background-color: #f0f5ff;
            }
            QPushButton#view_button_square {
                min-width: 60px;
                min-height: 60px;
                font-size: 10px;
                padding: 3px 5px;
                border-radius: 4px;
                background-color: #5d9cec;
                color: white;
            }
            QPushButton#view_button_square:hover { background-color: #4a89dc; }
            QPushButton#view_button_square:pressed { background-color: #3b7dd8; }
            QPushButton#view_button_square[active="true"] {
                background-color: #4caf50;
                font-weight: bold;
            }
            QSlider {
                min-height: 20px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #d1d9e6;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                height: 16px;
                margin: -5px 0;
                background: #4a6fa5;
                border-radius: 8px;
            }
        """)

    def setup_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Left panel
        left_panel = QFrame()
        left_panel.setFixedWidth(260)
        left_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # Title
        title_label = QLabel(self.title)
        title_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold; 
            color: #2c3e50;
            padding: 8px 0;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title_label)

        # Control panel
        self.control_panel = QGroupBox(self.lang_map[self.current_lang]['control_panel'])
        control_layout = QGridLayout()
        control_layout.setSpacing(8)
        control_layout.setContentsMargins(8, 10, 8, 8)

        self.loadModelButton = QPushButton(self.lang_map[self.current_lang]['load_model'])
        self.loadLabelsButton = QPushButton(self.lang_map[self.current_lang]['load_labels'])
        self.loadButton = QPushButton(self.lang_map[self.current_lang]['load_step'])
        self.clearButton = QPushButton(self.lang_map[self.current_lang]['clear'])
        self.exampleButton = QPushButton(self.lang_map[self.current_lang]['help'])
        self.viewCategoriesButton = QPushButton(self.lang_map[self.current_lang]['view_categories'])
        self.historyButton = QPushButton(self.lang_map[self.current_lang]['history'])
        self.exportImageButton = QPushButton(self.lang_map[self.current_lang]['export_image'])
        self.langButton = QPushButton("EN/中文" if self.current_lang == 'zh' else "中文/EN")

        for btn in [self.loadModelButton, self.loadLabelsButton, self.loadButton,
                    self.clearButton, self.exampleButton, self.viewCategoriesButton,
                    self.historyButton, self.exportImageButton, self.langButton]:
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(getattr(self, {
                self.lang_map[self.current_lang]['load_model']: "loadModel",
                self.lang_map[self.current_lang]['load_labels']: "loadLabelMapping",
                self.lang_map[self.current_lang]['load_step']: "loadSTEP",
                self.lang_map[self.current_lang]['clear']: "clearAll",
                self.lang_map[self.current_lang]['help']: "showExamples",
                self.lang_map[self.current_lang]['view_categories']: "showCategories",
                self.lang_map[self.current_lang]['history']: "showHistory",
                self.lang_map[self.current_lang]['export_image']: "exportImage",
                "EN/中文": "toggleLanguage",
                "中文/EN": "toggleLanguage"
            }[btn.text()]))

        control_layout.addWidget(self.loadModelButton, 0, 0, 1, 2)
        control_layout.addWidget(self.loadLabelsButton, 1, 0, 1, 2)
        control_layout.addWidget(self.loadButton, 2, 0, 1, 2)
        control_layout.addWidget(self.clearButton, 3, 0)
        control_layout.addWidget(self.exampleButton, 3, 1)
        control_layout.addWidget(self.viewCategoriesButton, 4, 0)
        control_layout.addWidget(self.historyButton, 4, 1)
        control_layout.addWidget(self.exportImageButton, 5, 0, 1, 2)
        control_layout.addWidget(self.langButton, 6, 0, 1, 2)
        self.control_panel.setLayout(control_layout)
        left_layout.addWidget(self.control_panel)

        # Display control panel
        self.display_panel = QGroupBox(self.lang_map[self.current_lang]['display_control'])
        display_layout = QVBoxLayout()
        display_layout.setSpacing(8)
        display_layout.setContentsMargins(8, 10, 8, 8)

        # Display mode
        self.mode_group = QGroupBox(self.lang_map[self.current_lang]['display_mode'])
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(6)
        mode_layout.setContentsMargins(5, 5, 5, 5)
        self.shadedButton = QPushButton(self.lang_map[self.current_lang]['shaded'])
        self.wireframeButton = QPushButton(self.lang_map[self.current_lang]['wireframe'])
        self.shadedButton.setCheckable(True)
        self.wireframeButton.setCheckable(True)
        self.shadedButton.setChecked(True)
        self.shadedButton.clicked.connect(self.setDisplayMode)
        self.wireframeButton.clicked.connect(self.setDisplayMode)
        mode_layout.addWidget(self.shadedButton)
        mode_layout.addWidget(self.wireframeButton)
        self.mode_group.setLayout(mode_layout)
        display_layout.addWidget(self.mode_group)

        # Transparency control
        self.transparency_group = QGroupBox(self.lang_map[self.current_lang]['transparency'])
        transparency_layout = QVBoxLayout()
        transparency_layout.setContentsMargins(5, 5, 5, 5)
        self.transparencySlider = QSlider(Qt.Horizontal)
        self.transparencySlider.setRange(0, 100)
        self.transparencySlider.setValue(0)
        self.transparencySlider.valueChanged.connect(self.setTransparency)
        transparency_layout.addWidget(self.transparencySlider)
        self.transparency_group.setLayout(transparency_layout)
        display_layout.addWidget(self.transparency_group)

        # Color control
        self.color_group = QGroupBox(self.lang_map[self.current_lang]['color'])
        color_layout = QHBoxLayout()
        color_layout.setSpacing(6)
        color_layout.setContentsMargins(5, 5, 5, 5)
        self.customColorButton = QPushButton(self.lang_map[self.current_lang]['choose_color'])
        self.customColorButton.clicked.connect(self.chooseCustomColor)
        color_layout.addWidget(self.customColorButton)
        self.color_group.setLayout(color_layout)
        display_layout.addWidget(self.color_group)

        self.display_panel.setLayout(display_layout)
        left_layout.addWidget(self.display_panel)

        # View control panel
        self.view_control_panel = QGroupBox(self.lang_map[self.current_lang]['view_control'])
        view_layout = QGridLayout()
        view_layout.setSpacing(8)
        view_layout.setContentsMargins(8, 10, 8, 8)

        self.isoViewButton = QPushButton(self.lang_map[self.current_lang]['iso_view'])
        self.resetViewButton = QPushButton(self.lang_map[self.current_lang]['reset_view'])

        for btn in [self.isoViewButton, self.resetViewButton]:
            btn.setCursor(Qt.PointingHandCursor)
            btn.setObjectName("view_button_square")
            btn.clicked.connect(partial(self.setView, btn.text().split("\n")[0]))

        view_layout.addWidget(self.isoViewButton, 0, 0, 1, 2)
        view_layout.addWidget(self.resetViewButton, 1, 0, 1, 2)
        view_layout.setRowStretch(2, 1)

        self.view_control_panel.setLayout(view_layout)
        left_layout.addWidget(self.view_control_panel)
        left_layout.addStretch(1)

        # Right panel
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # 3D display area
        display_frame = QFrame()
        display_frame.setObjectName("display_frame")
        display_layout = QHBoxLayout(display_frame)
        display_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = qtViewer3d(self)
        self.canvas.setMinimumSize(700, 500)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMouseTracking(True)

        self.drop_label = QLabel(self.lang_map[self.current_lang]['drop_label'])
        self.drop_label.setObjectName("drop_label")
        self.drop_label.setVisible(False)

        display_layout.addWidget(self.canvas)
        display_layout.addWidget(self.drop_label)
        right_layout.addWidget(display_frame, 1)

        # Status bar
        status_frame = QFrame()
        status_layout = QVBoxLayout(status_frame)
        status_layout.setContentsMargins(6, 6, 6, 6)

        self.predictionLabel = QLabel("")
        self.predictionLabel.setObjectName("status_label")
        self.predictionLabel.setAlignment(Qt.AlignCenter)
        self.predictionLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.info_label = QLabel(self.lang_map[self.current_lang]['copyright'])
        self.info_label.setAlignment(Qt.AlignRight)
        self.info_label.setStyleSheet("color: #666; font-size: 9px;")

        status_layout.addWidget(self.predictionLabel)
        status_layout.addWidget(self.info_label)
        right_layout.addWidget(status_frame)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        self.setLayout(main_layout)

    def convert_step_to_bin(self, step_path, bin_path):
        """Convert STEP file to DGL graph binary format"""
        try:
            # Load the STEP file
            solid = load_step(step_path)[0]

            # Build the graph with default parameters
            graph = build_graph(solid,
                                curv_num_u_samples=10,
                                surf_num_u_samples=10,
                                surf_num_v_samples=10)

            # Save the graph
            dgl.data.utils.save_graphs(str(bin_path), [graph])
            return True
        except Exception as e:
            self.showError(self.lang_map[self.current_lang]['convert_error'].format(str(e)))
            return False

    def processSTEPFile(self, file_path):
        lang = self.lang_map[self.current_lang]
        if not self.model_loaded:
            self.showError(lang['no_model_error'])
            return
        if not self.labels_loaded:
            self.showError(lang['no_labels_error'])
            return

        try:
            # 1. Display the STEP file
            shapes = read_step_file(file_path)
            ais = self.display.DisplayShape(shapes, update=True)[0]
            self.display.FitAll()
            self.ais_list.append(ais)

            # 2. Convert to BIN format in temp directory
            self.updateStatus(lang['converting'])
            QApplication.processEvents()  # Update UI

            bin_filename = os.path.basename(file_path) + ".bin"
            bin_path = os.path.join(self.temp_dir, bin_filename)

            if not self.convert_step_to_bin(file_path, bin_path):
                return

            # 3. Classify using the model
            self.updateStatus(lang['classifying'])
            QApplication.processEvents()

            # Load the preprocessor and model
            from preprocessor import load_one_graph
            from feature_extractor import init

            sample = load_one_graph(bin_path)
            inputs = sample["graph"]
            inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
            inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)

            with torch.no_grad():
                logits = init(bin_path, self.current_model)
                preds = F.softmax(logits, dim=-1)
                max_index = torch.argmax(preds, dim=-1).item()
                confidence = torch.max(preds).item() * 100

            prediction_text = self.label_mapping.get(str(max_index), f'Class {max_index}')
            result_text = lang['result_text'].format(prediction_text, confidence)
            self.updateStatus(result_text)

            self.addToHistory(file_path, prediction_text, confidence)
            self.step_loaded = True
            self.loadButton.setProperty("loaded", "true")
            self.loadButton.style().polish(self.loadButton)

        except Exception as e:
            self.showError(f"{lang['process_error'].format(str(e))}")

    def toggleLanguage(self):
        self.current_lang = 'en' if self.current_lang == 'zh' else 'zh'
        self.updateUI()

    def updateUI(self):
        lang = self.lang_map[self.current_lang]

        # Update window title
        self.setWindowTitle(lang['title'])

        # Update button texts
        self.loadModelButton.setText(lang['load_model'])
        self.loadLabelsButton.setText(lang['load_labels'])
        self.loadButton.setText(lang['load_step'])
        self.clearButton.setText(lang['clear'])
        self.exampleButton.setText(lang['help'])
        self.viewCategoriesButton.setText(lang['view_categories'])
        self.historyButton.setText(lang['history'])
        self.exportImageButton.setText(lang['export_image'])
        self.langButton.setText("EN/中文" if self.current_lang == 'zh' else "中文/EN")

        # Update group box titles
        self.control_panel.setTitle(lang['control_panel'])
        self.display_panel.setTitle(lang['display_control'])
        self.mode_group.setTitle(lang['display_mode'])
        self.shadedButton.setText(lang['shaded'])
        self.wireframeButton.setText(lang['wireframe'])
        self.transparency_group.setTitle(lang['transparency'])
        self.color_group.setTitle(lang['color'])
        self.customColorButton.setText(lang['choose_color'])
        self.view_control_panel.setTitle(lang['view_control'])
        self.isoViewButton.setText(lang['iso_view'])
        self.resetViewButton.setText(lang['reset_view'])
        self.drop_label.setText(lang['drop_label'])

        # Update bottom info
        self.info_label.setText(lang['copyright'])

        # Update history window
        if hasattr(self, 'history_window') and self.history_window:
            self.history_window.setWindowTitle(lang['history_title'])
            if hasattr(self, 'clear_btn'):
                self.clear_btn.setText(lang['clear_history'])

    def exportImage(self):
        lang = self.lang_map[self.current_lang]
        if not self.ais_list:
            self.showError(lang['no_model'])
            return

        # Create preview dialog
        preview_dialog = QDialog(self)
        preview_dialog.setWindowTitle(lang['preview_title'])
        preview_dialog.setMinimumSize(800, 600)

        layout = QVBoxLayout(preview_dialog)

        # Preview label
        preview_label = QLabel()
        preview_label.setAlignment(Qt.AlignCenter)
        preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Zoom control
        zoom_group = QGroupBox(lang['zoom_control'])
        zoom_layout = QHBoxLayout()

        zoom_slider = QSlider(Qt.Horizontal)
        zoom_slider.setRange(50, 200)  # 50% to 200%
        zoom_slider.setValue(100)

        zoom_value = QLabel("100%")
        zoom_value.setFixedWidth(50)

        zoom_layout.addWidget(QLabel(lang['zoom']))
        zoom_layout.addWidget(zoom_slider)
        zoom_layout.addWidget(zoom_value)
        zoom_group.setLayout(zoom_layout)

        # Button area
        button_layout = QHBoxLayout()
        save_button = QPushButton(lang['save_image'])
        cancel_button = QPushButton(lang['cancel'])

        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)

        # Add to main layout
        layout.addWidget(preview_label)
        layout.addWidget(zoom_group)
        layout.addLayout(button_layout)

        # Function to generate preview
        def generate_preview(zoom_factor=1.0):
            try:
                # Create temp file
                temp_file = os.path.join(self.temp_dir, "temp_preview.png")

                view = self.display.GetView()

                # Calculate combined bounding box
                bbox = Bnd_Box()
                for ais in self.ais_list:
                    brepbndlib.AddOptimal(ais.Shape(), bbox, True, True)

                if not bbox.IsVoid():
                    view.FitAll(bbox, 0.01)
                    current_scale = view.Scale()
                    view.SetScale(current_scale * zoom_factor)
                    view.ZFitAll(0.01)

                # Take screenshot
                view.Dump(temp_file)

                # Restore view
                if not bbox.IsVoid():
                    view.FitAll()

                # Show preview
                pixmap = QPixmap(temp_file)
                preview_label.setPixmap(pixmap.scaled(
                    preview_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))

                # Delete temp file
                os.remove(temp_file)

            except Exception as e:
                self.showError(f"{lang['process_error'].format(str(e))}")

        # Zoom slider event
        def on_zoom_changed(value):
            zoom_factor = value / 100.0
            zoom_value.setText(f"{value}%")
            generate_preview(zoom_factor)

        zoom_slider.valueChanged.connect(on_zoom_changed)

        # Button events
        def on_save():
            options = QFileDialog.Options()
            fileName, _ = QFileDialog.getSaveFileName(
                preview_dialog, lang['save_image'], "",
                "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg)",
                options=options
            )

            if fileName:
                try:
                    zoom_factor = zoom_slider.value() / 100.0
                    view = self.display.GetView()

                    # Calculate combined bounding box
                    bbox = Bnd_Box()
                    for ais in self.ais_list:
                        brepbndlib.AddOptimal(ais.Shape(), bbox, True, True)

                    if not bbox.IsVoid():
                        view.FitAll(bbox, 0.01)
                        current_scale = view.Scale()
                        view.SetScale(current_scale * zoom_factor)
                        view.ZFitAll(0.01)

                    # Take screenshot
                    view.Dump(fileName)

                    # Restore view
                    if not bbox.IsVoid():
                        view.FitAll()

                    self.updateStatus(lang['image_saved'].format(os.path.basename(fileName)))
                    preview_dialog.accept()

                except Exception as e:
                    self.showError(f"{lang['process_error'].format(str(e))}")

        save_button.clicked.connect(on_save)
        cancel_button.clicked.connect(preview_dialog.reject)

        # Initial preview generation
        generate_preview()

        # Show dialog
        preview_dialog.exec_()

    def setDisplayMode(self):
        if not self.ais_list:
            return

        sender = self.sender()
        if sender == self.shadedButton:
            self.shadedButton.setChecked(True)
            self.wireframeButton.setChecked(False)
            display_mode = AIS_Shaded
        else:
            self.shadedButton.setChecked(False)
            self.wireframeButton.setChecked(True)
            display_mode = AIS_WireFrame

        for ais in self.ais_list:
            self.display.Context.SetDisplayMode(ais, display_mode, True)
        self.display.Context.UpdateCurrentViewer()

    def setTransparency(self, value):
        if not self.ais_list:
            return

        transparency = value / 100.0
        for ais in self.ais_list:
            self.display.Context.SetTransparency(ais, transparency, True)
        self.display.Context.UpdateCurrentViewer()

    def setShapeColor(self, color):
        if not self.ais_list:
            return

        for ais in self.ais_list:
            self.display.Context.SetColor(ais, color, True)
        self.display.Context.UpdateCurrentViewer()

    def chooseCustomColor(self):
        color = QColorDialog.getColor()
        if color.isValid():
            r = color.red() / 255.0
            g = color.green() / 255.0
            b = color.blue() / 255.0
            custom_color = Quantity_Color(r, g, b, Quantity_TOC_RGB)
            self.setShapeColor(custom_color)

    def setView(self, view_name):
        if not hasattr(self, 'display'):
            return

        for btn in [self.isoViewButton, self.resetViewButton]:
            btn.setProperty("active", "false")
            btn.style().polish(btn)

        current_btn = {
            self.lang_map[self.current_lang]['iso_view'].split("\n")[0]: self.isoViewButton,
            self.lang_map[self.current_lang]['reset_view'].split("\n")[0]: self.resetViewButton
        }.get(view_name)

        if current_btn:
            current_btn.setProperty("active", "true")
            current_btn.style().polish(current_btn)

        view = self.display.GetView()
        viewer = self.display.GetViewer()

        if view_name == self.lang_map[self.current_lang]['reset_view'].split("\n")[0]:
            view.Reset()
            if self.ais_list:
                self.display.FitAll()
            view.Redraw()
            return

        view.Reset()

        if view_name == self.lang_map[self.current_lang]['iso_view'].split("\n")[0]:
            view.SetProj(1, 1, 1)
            view.SetUp(0, 1, 0)

        if self.ais_list:
            try:
                bbox = Bnd_Box()
                for ais in self.ais_list:
                    shape = ais.Shape()
                    brepbndlib.Add(shape, bbox, False)

                if not bbox.IsVoid():
                    center_x = int((bbox.CornerMin().X() + bbox.CornerMax().X()) / 2)
                    center_y = int((bbox.CornerMin().Y() + bbox.CornerMax().Y()) / 2)
                    view.SetCenter(center_x, center_y)
            except Exception as e:
                print(f"Error setting view center: {str(e)}")

        view.Redraw()
        self.display.FitAll()

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key_I:
            self.setView(self.lang_map[self.current_lang]['iso_view'].split("\n")[0])
        elif key == Qt.Key_Escape:
            self.setView(self.lang_map[self.current_lang]['reset_view'].split("\n")[0])
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton or event.button() == Qt.RightButton:
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if not self.last_mouse_pos:
            return

        delta = event.pos() - self.last_mouse_pos
        self.last_mouse_pos = event.pos()

        view = self.display.GetView()

        if event.buttons() & Qt.LeftButton:
            view.Rotate(delta.x(), delta.y(), 0, 0, 1, 0)
        elif event.buttons() & Qt.RightButton:
            view.Pan(delta.x(), -delta.y(), 0)

        view.Redraw()

    def mouseReleaseEvent(self, event):
        self.last_mouse_pos = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        view = self.display.GetView()
        if delta > 0:
            view.SetZoom(1.1)
        else:
            view.SetZoom(0.9)
        view.Redraw()

    def showHistory(self):
        lang = self.lang_map[self.current_lang]
        if not self.history:
            QMessageBox.information(self, lang['history_title'], lang['no_history'])
            return

        if self.history_window is None:
            self.history_window = QWidget()
            self.history_window.setWindowTitle(lang['history_title'])
            self.history_window.setMinimumSize(500, 400)

            layout = QVBoxLayout()

            self.clear_btn = QPushButton(lang['clear_history'])
            self.clear_btn.clicked.connect(self.clearHistory)
            layout.addWidget(self.clear_btn)

            self.history_list = QListWidget()
            self.history_list.itemDoubleClicked.connect(self.loadFromHistory)
            layout.addWidget(self.history_list)

            self.history_window.setLayout(layout)

        self.history_list.clear()
        for item in reversed(self.history):
            self.history_list.addItem(
                f"{item['time']} - {item['filename']}: {item['prediction']} ({item['confidence']:.1f}%)"
            )

        self.history_window.show()

    def clearHistory(self):
        lang = self.lang_map[self.current_lang]
        self.history = []
        if self.history_window:
            self.history_list.clear()
        QMessageBox.information(self, lang['history_title'], lang['history_cleared'])

    def loadFromHistory(self, item):
        index = len(self.history) - 1 - self.history_list.row(item)
        history_item = self.history[index]

        try:
            self.clearDisplay()
            shapes = read_step_file(history_item['filepath'])
            ais = self.display.DisplayShape(shapes, update=True)[0]
            self.display.FitAll()
            self.ais_list.append(ais)

            lang = self.lang_map[self.current_lang]
            result_text = lang['result_text'].format(history_item['prediction'], history_item['confidence'])
            self.updateStatus(result_text)

        except Exception as e:
            self.showError(f"{self.lang_map[self.current_lang]['process_error'].format(str(e))}")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_label.setStyleSheet("background-color: rgba(234, 242, 255, 0.9);")
            self.drop_label.setVisible(True)

    def dragLeaveEvent(self, event):
        self.drop_label.setStyleSheet("")
        self.drop_label.setVisible(False)

    def dropEvent(self, event):
        self.drop_label.setStyleSheet("")
        self.drop_label.setVisible(False)
        urls = event.mimeData().urls()
        if not urls:
            return

        file_path = urls[0].toLocalFile()
        if not file_path:
            return

        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.step', '.stp']:
            self.handleDroppedSTEP(file_path)
        elif ext in ['.ckpt', '.pt', '.pth']:
            self.handleDroppedModel(file_path)
        elif ext == '.json':
            self.handleDroppedLabels(file_path)
        else:
            self.showError(self.lang_map[self.current_lang]['unsupported_file'])

    def handleDroppedModel(self, file_path):
        lang = self.lang_map[self.current_lang]
        self.current_model = file_path
        self.model_loaded = True
        self.loadModelButton.setProperty("loaded", "true")
        self.loadModelButton.style().polish(self.loadModelButton)
        self.updateStatus(lang['model_loaded'].format(os.path.basename(file_path)))

    def handleDroppedLabels(self, file_path):
        lang = self.lang_map[self.current_lang]
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.label_mapping = json.load(f)
            self.labels_loaded = True
            self.loadLabelsButton.setProperty("loaded", "true")
            self.loadLabelsButton.style().polish(self.loadLabelsButton)
            self.updateStatus(lang['labels_loaded'].format(os.path.basename(file_path)))
        except Exception as e:
            self.showError(f"{lang['load_error'].format(str(e))}")

    def handleDroppedSTEP(self, file_path):
        lang = self.lang_map[self.current_lang]
        self.processSTEPFile(file_path)

    def loadModel(self):
        lang = self.lang_map[self.current_lang]
        fileName, _ = QFileDialog.getOpenFileName(self, lang['load_model'], "", "Model Files (*.ckpt *.pt *.pth)")
        if fileName:
            self.handleDroppedModel(fileName)

    def loadLabelMapping(self):
        lang = self.lang_map[self.current_lang]
        fileName, _ = QFileDialog.getOpenFileName(self, lang['load_labels'], "", "JSON Files (*.json)")
        if fileName:
            self.handleDroppedLabels(fileName)

    def loadSTEP(self):
        lang = self.lang_map[self.current_lang]
        fileName, _ = QFileDialog.getOpenFileName(
            self, lang['load_step'], "",
            "STEP Files (*.step *.STEP *.stp *.STP)"
        )
        if fileName:
            self.handleDroppedSTEP(fileName)

    def clearDisplay(self):
        for ais in self.ais_list:
            self.display.Context.Erase(ais, True)
        self.ais_list = []
        self.display.FitAll()

    def clearAll(self):
        self.clearDisplay()
        self.current_model = None
        self.label_mapping = None
        self.model_loaded = False
        self.labels_loaded = False
        self.step_loaded = False

        for btn in [self.loadModelButton, self.loadLabelsButton, self.loadButton]:
            btn.setProperty("loaded", "false")
            btn.style().polish(btn)

        self.updateStatus("")

    def showExamples(self):
        help_texts = {
            'zh': """
            <b>使用说明:</b>
            <ol>
                <li>先加载模型文件(.ckpt/.pt/.pth)</li>
                <li>再加载标签映射文件(.json)</li>
                <li>最后加载STEP文件进行预测</li>
            </ol>
            <b>支持拖拽操作</b>
            <p><b>视图控制:</b></p>
            <ul>
                <li>等轴视图(I): 标准3D等轴测视图</li>
                <li>重置视图(ESC): 重置视图到默认位置</li>
            </ul>
            <p><b>显示控制:</b></p>
            <ul>
                <li>着色/线框: 切换模型的显示模式</li>
                <li>透明度: 调整模型的透明度</li>
                <li>颜色: 更改模型的显示颜色</li>
            </ul>
            <p><b>鼠标控制:</b></p>
            <ul>
                <li>左键拖动: 旋转视图</li>
                <li>右键拖动: 平移视图</li>
                <li>滚轮: 缩放视图</li>
            </ul>""",
            'en': """
            <b>Instructions:</b>
            <ol>
                <li>First load model file (.ckpt/.pt/.pth)</li>
                <li>Then load label mapping file (.json)</li>
                <li>Finally load STEP file for prediction</li>
            </ol>
            <b>Drag and drop supported</b>
            <p><b>View Control:</b></p>
            <ul>
                <li>Isometric View(I): Standard 3D isometric view</li>
                <li>Reset View(ESC): Reset view to default position</li>
            </ul>
            <p><b>Display Control:</b></p>
            <ul>
                <li>Shaded/Wireframe: Toggle display mode</li>
                <li>Transparency: Adjust model transparency</li>
                <li>Color: Change model display color</li>
            </ul>
            <p><b>Mouse Control:</b></p>
            <ul>
                <li>Left drag: Rotate view</li>
                <li>Right drag: Pan view</li>
                <li>Wheel: Zoom view</li>
            </ul>"""
        }

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(self.lang_map[self.current_lang]['help'])
        msg.setTextFormat(Qt.RichText)
        msg.setText(help_texts[self.current_lang])
        msg.exec_()

    def showCategories(self):
        lang = self.lang_map[self.current_lang]
        if not self.label_mapping:
            self.showError(lang['no_labels_error'])
            return

        categories = lang['categories_text'].format("<br>".join(
            f"{k}. {v}" for k, v in sorted(self.label_mapping.items(), key=lambda x: int(x[0]))))

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(lang['categories_title'])
        msg.setTextFormat(Qt.RichText)
        msg.setText(categories)
        msg.exec_()

    def force_refresh_display(self):
        self.canvas.resize(self.canvas.width() + 1, self.canvas.height() + 1)
        self.canvas.resize(self.canvas.width() - 1, self.canvas.height() - 1)
        self.display.Repaint()

    def updateStatus(self, message, is_error=False):
        self.predictionLabel.setText(message)
        self.predictionLabel.setProperty("error", "true" if is_error else "false")
        self.predictionLabel.style().polish(self.predictionLabel)

    def showError(self, message):
        lang = self.lang_map[self.current_lang]
        self.updateStatus(message, is_error=True)
        QMessageBox.critical(self, lang['error'], message)

    def addToHistory(self, filepath, prediction, confidence):
        filename = os.path.basename(filepath)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.history.append({
            'time': timestamp,
            'filename': filename,
            'filepath': filepath,
            'prediction': prediction,
            'confidence': confidence
        })

        if len(self.history) > 50:
            self.history = self.history[-50:]

    def closeEvent(self, event):
        """Clean up temp directory on exit"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont("Microsoft YaHei", 9))
    window = App()
    window.show()
    sys.exit(app.exec_())