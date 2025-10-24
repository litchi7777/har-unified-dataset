#!/usr/bin/env python3
"""
センサデータ可視化のためのWebアプリケーション

階層的なナビゲーションでデータセットを探索できます：
1. トップページ：データセット一覧
2. データセットページ：ユーザー×センサー×モダリティのグリッド
3. 可視化ページ：選択したデータの可視化

使用例:
    python serve.py
    python serve.py --port 8080
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from flask import Flask, render_template_string, request, jsonify

# Plotlyのインポート
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Error: plotly is not installed. Please install it:")
    print("  pip install plotly")
    sys.exit(1)

# データセット情報のインポート
sys.path.insert(0, str(Path(__file__).parent))
from src.dataset_info import DATASETS

app = Flask(__name__)

# グローバル設定
DATA_DIR = Path('data/processed')

# データセット名からアクティビティ名を取得するヘルパー関数
def get_activity_name(dataset_name: str, class_id: int) -> str:
    """データセット名とクラスIDから行動名を取得"""
    dataset_key = dataset_name.upper()
    if dataset_key in DATASETS and 'labels' in DATASETS[dataset_key]:
        return DATASETS[dataset_key]['labels'].get(class_id, f'Class {class_id}')
    return f'Class {class_id}'

# HTMLテンプレート
NEW_UI_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HAR Data Visualization - Interactive</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            overflow: hidden;
            height: 100vh;
            width: 100vw;
        }
        .app-container {
            display: flex;
            height: 100vh;
            width: 100vw;
            max-width: 100vw;
            overflow: hidden;
        }

        /* 最左ナビゲーションバー */
        .nav-bar {
            width: 80px;
            background: #202124;
            display: flex;
            flex-direction: column;
            border-right: 1px solid #000;
        }
        .nav-item {
            padding: 20px 0;
            text-align: center;
            cursor: pointer;
            color: #9aa0a6;
            border-left: 3px solid transparent;
            transition: all 0.2s;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
        }
        .nav-item:hover {
            background: #292a2d;
            color: #e8eaed;
        }
        .nav-item.active {
            background: #292a2d;
            color: #1a73e8;
            border-left-color: #1a73e8;
        }
        .nav-icon {
            font-size: 24px;
        }
        .nav-label {
            font-size: 11px;
            font-weight: 500;
        }

        /* 左サイドバー */
        .sidebar {
            width: 300px;
            min-width: 250px;
            max-width: 400px;
            background: white;
            border-right: 1px solid #e0e0e0;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .sidebar-header {
            padding: 16px;
            border-bottom: 1px solid #e0e0e0;
            background: #1a73e8;
            color: white;
        }
        .sidebar-header h1 {
            font-size: 18px;
            font-weight: 500;
        }
        .tree-container {
            flex: 1;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 8px;
        }
        .tree-node {
            padding: 8px;
            cursor: pointer;
            border-radius: 4px;
            margin: 2px 0;
            user-select: none;
            -webkit-tap-highlight-color: transparent;
        }
        .tree-node:hover {
            background: #f0f0f0;
        }
        .tree-node:focus {
            outline: none;
        }
        .tree-node.dataset {
            font-weight: 500;
            color: #202124;
        }
        .tree-node.user {
            margin-left: 8px;
            color: #5f6368;
        }
        .tree-node.position {
            margin-left: 16px;
            color: #1a73e8;
        }
        .tree-node.modality {
            margin-left: 24px;
            color: #fb8c00;
        }
        .tree-node.class {
            margin-left: 32px;
            color: #34a853;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
        }
        .add-btn {
            background: #1a73e8;
            color: white;
            border: none;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 11px;
            cursor: pointer;
            flex-shrink: 0;
        }
        .add-btn:hover {
            background: #1557b0;
        }
        .collapsed > .tree-children {
            display: none;
        }

        /* 右側メインエリア */
        .main-area {
            flex: 1;
            min-width: 0;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .main-header {
            padding: 16px 24px;
            background: white;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .main-header h2 {
            font-size: 16px;
            font-weight: 500;
            color: #202124;
        }
        .header-controls {
            display: flex;
            gap: 12px;
            align-items: center;
        }
        .stats-btn {
            padding: 8px 16px;
            border: 1px solid #1a73e8;
            border-radius: 4px;
            background: white;
            color: #1a73e8;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: background 0.2s, color 0.2s;
        }
        .stats-btn:hover {
            background: #1a73e8;
            color: white;
        }
        .panels-container {
            flex: 1;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 16px;
            max-width: 100%;
        }

        /* パネル */
        .panel {
            background: white;
            border: 1px solid #dadce0;
            border-radius: 8px;
            margin-bottom: 16px;
            overflow: hidden;
            max-width: 100%;
        }
        .panel-header {
            padding: 12px 16px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .panel-title {
            font-weight: 500;
            color: #202124;
            font-size: 14px;
        }
        .panel-controls {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .panel-controls select,
        .panel-controls input {
            padding: 4px 8px;
            border: 1px solid #dadce0;
            border-radius: 4px;
            font-size: 12px;
        }
        .panel-controls button {
            padding: 4px 12px;
            border: 1px solid #dadce0;
            border-radius: 4px;
            background: white;
            cursor: pointer;
            font-size: 12px;
        }
        .panel-controls button:hover {
            background: #f8f9fa;
        }
        .remove-btn {
            color: #d93025;
            border-color: #d93025;
        }
        .remove-btn:hover {
            background: #fce8e6;
        }
        .panel-content {
            display: flex;
            max-width: 100%;
            overflow: hidden;
        }
        .panel-metadata {
            min-width: 120px;
            max-width: 150px;
            padding: 12px;
            border-right: 1px solid #e0e0e0;
            background: #fafafa;
            flex-shrink: 0;
        }
        .metadata-item {
            margin-bottom: 8px;
            font-size: 12px;
        }
        .metadata-label {
            color: #5f6368;
            font-size: 10px;
            text-transform: uppercase;
            margin-bottom: 2px;
        }
        .metadata-value {
            color: #202124;
            font-weight: 500;
        }
        .panel-plots {
            flex: 1;
            min-width: 0;
            display: flex;
            overflow-x: auto;
            overflow-y: hidden;
            padding: 12px;
            gap: 12px;
        }
        .plot-item {
            min-width: 280px;
            max-width: 500px;
            flex-shrink: 0;
            height: 250px;
            display: flex;
            flex-direction: column;
        }
        .plot-header {
            font-size: 12px;
            color: #5f6368;
            margin-bottom: 8px;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 4px;
            flex-shrink: 0;
        }
        .plot-item > div:last-child {
            flex: 1;
            min-height: 0;
        }
        .empty-state {
            text-align: center;
            padding: 48px;
            color: #5f6368;
        }
        .empty-state h3 {
            margin-bottom: 8px;
            color: #202124;
        }

        /* スクロールバー */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        ::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #a1a1a1;
        }

        .loading {
            text-align: center;
            padding: 24px;
            color: #5f6368;
        }

        /* 統計モーダル */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 2000;
            align-items: center;
            justify-content: center;
        }
        .modal.show {
            display: flex;
        }
        .modal-content {
            background: white;
            border-radius: 8px;
            max-width: 900px;
            max-height: 80vh;
            width: 90%;
            display: flex;
            flex-direction: column;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .modal-header {
            padding: 20px 24px;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .modal-title {
            font-size: 20px;
            font-weight: 500;
            color: #202124;
        }
        .modal-close {
            background: none;
            border: none;
            font-size: 24px;
            color: #5f6368;
            cursor: pointer;
            padding: 0;
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
        }
        .modal-close:hover {
            background: #f0f0f0;
        }
        .modal-body {
            padding: 24px;
            overflow-y: auto;
            flex: 1;
        }
        .stats-loading {
            text-align: center;
            padding: 48px;
            color: #5f6368;
        }
        .stats-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .stats-card {
            background: #f8f9fa;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }
        .stats-card-label {
            font-size: 12px;
            color: #5f6368;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .stats-card-value {
            font-size: 32px;
            font-weight: 500;
            color: #1a73e8;
        }
        .stats-dataset {
            margin-bottom: 24px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }
        .stats-dataset-header {
            background: #1a73e8;
            color: white;
            padding: 12px 16px;
            font-weight: 500;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .stats-dataset-header:hover {
            background: #1557b0;
        }
        .stats-dataset-body {
            padding: 16px;
            background: white;
        }
        .stats-dataset.collapsed .stats-dataset-body {
            display: none;
        }
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        .stats-table th {
            text-align: left;
            padding: 8px;
            background: #f8f9fa;
            color: #5f6368;
            font-weight: 500;
            border-bottom: 2px solid #e0e0e0;
        }
        .stats-table td {
            padding: 8px;
            border-bottom: 1px solid #f0f0f0;
            color: #202124;
        }
        .stats-table tr:hover {
            background: #f8f9fa;
        }
        .expand-icon {
            transition: transform 0.2s;
        }
        .stats-dataset.collapsed .expand-icon {
            transform: rotate(-90deg);
        }

        /* 統計ビュー専用スタイル */
        .stats-view-container {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            background: #f5f5f5;
        }
        .stats-section {
            background: white;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid #dadce0;
        }
        .stats-section-title {
            font-size: 18px;
            font-weight: 500;
            color: #202124;
            margin-bottom: 16px;
        }

        /* 表示切り替え */
        .view-mode {
            display: none !important;
            flex: 1;
        }
        .view-mode.active {
            display: flex !important;
        }
        #statsView.active {
            flex-direction: column;
        }

        /* レスポンシブデザイン */
        @media (max-width: 1200px) {
            .sidebar {
                width: 250px;
                min-width: 200px;
            }
            .plot-item {
                min-width: 250px;
            }
        }

        @media (max-width: 768px) {
            .nav-bar {
                width: 60px;
            }
            .nav-label {
                display: none;
            }
            .sidebar {
                width: 200px;
                min-width: 150px;
            }
            .panel-metadata {
                min-width: 100px;
                max-width: 120px;
            }
            .plot-item {
                min-width: 220px;
                max-width: 400px;
            }
            .main-header {
                padding: 12px 16px;
            }
            .panels-container {
                padding: 12px;
            }
        }

        @media (max-width: 480px) {
            .nav-bar {
                display: none;
            }
            .sidebar {
                width: 100%;
                max-width: 100%;
                border-right: none;
            }
            .main-area {
                display: none;
            }
            .app-container {
                flex-direction: column;
            }
        }

        /* Plotly グラフのレスポンシブ対応 */
        .js-plotly-plot {
            width: 100%;
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- 最左ナビゲーションバー -->
        <div class="nav-bar">
            <div class="nav-item active" onclick="switchView('data')" id="navData">
                <div class="nav-icon">📊</div>
                <div class="nav-label">Data</div>
            </div>
            <div class="nav-item" onclick="switchView('stats')" id="navStats">
                <div class="nav-icon">📈</div>
                <div class="nav-label">Statistics</div>
            </div>
        </div>

        <!-- Data View -->
        <div id="dataView" class="view-mode active" style="display: flex; flex: 1;">
            <!-- 左サイドバー -->
            <div class="sidebar">
                <div class="sidebar-header">
                    <h1>HAR Datasets</h1>
                </div>
                <div class="tree-container" id="treeContainer">
                    <div class="loading">Loading...</div>
                </div>
            </div>

            <!-- 右側メインエリア -->
            <div class="main-area">
                <div class="main-header">
                    <h2>Sensor Data Panels</h2>
                    <div class="header-controls">
                        <label style="font-size: 14px; color: #5f6368;">Sampling:</label>
                        <select id="samplingMode" onchange="updateSamplingMode()" style="padding: 6px 12px; border: 1px solid #dadce0; border-radius: 4px; font-size: 14px;">
                            <option value="random">Random</option>
                            <option value="sequential">Sequential</option>
                        </select>
                        <button onclick="clearAllPanels()" style="padding: 8px 16px; border: 1px solid #dadce0; border-radius: 4px; background: white; cursor: pointer;">
                            Clear All
                        </button>
                    </div>
                </div>
                <div class="panels-container" id="panelsContainer">
                    <div class="empty-state">
                        <h3>No panels added</h3>
                        <p>Select a sensor from the left sidebar to get started</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Statistics View -->
        <div id="statsView" class="view-mode">
            <div class="main-header" style="border-bottom: 1px solid #e0e0e0;">
                <h2>Dataset Statistics</h2>
                <div class="header-controls">
                    <button onclick="loadStatistics()" style="padding: 8px 16px; border: 1px solid #1a73e8; border-radius: 4px; background: white; color: #1a73e8; cursor: pointer; font-weight: 500;">
                        Refresh
                    </button>
                </div>
            </div>
            <div class="stats-view-container" id="statsViewContainer">
                <div class="stats-loading">Loading statistics...</div>
            </div>
        </div>
    </div>

    <script>
        let panels = [];
        let panelIdCounter = 0;
        let globalSamplingMode = 'random';  // グローバルなサンプリングモード
        let currentView = 'data';  // 現在のビュー
        let statsCache = null;  // 統計情報のキャッシュ

        // ビュー切り替え
        function switchView(viewName) {
            // ナビゲーションアイテムの更新
            document.querySelectorAll('.nav-item').forEach(item => {
                item.classList.remove('active');
            });
            document.getElementById('nav' + viewName.charAt(0).toUpperCase() + viewName.slice(1)).classList.add('active');

            // ビューの切り替え
            document.querySelectorAll('.view-mode').forEach(view => {
                view.classList.remove('active');
            });
            document.getElementById(viewName + 'View').classList.add('active');

            currentView = viewName;

            // 統計ビューに切り替えた時は統計情報を読み込む
            if (viewName === 'stats' && !statsCache) {
                loadStatistics();
            }
        }

        // 統計情報を読み込み
        async function loadStatistics() {
            const container = document.getElementById('statsViewContainer');
            container.innerHTML = '<div class="stats-loading">Loading statistics...</div>';

            try {
                const response = await fetch('/api/statistics');
                const stats = await response.json();

                if (stats.error) {
                    container.innerHTML = `<div class="stats-loading">Error: ${stats.error}</div>`;
                    return;
                }

                statsCache = stats;
                renderStatisticsView(stats);
            } catch (error) {
                console.error('Failed to load statistics:', error);
                container.innerHTML = '<div class="stats-loading">Failed to load statistics</div>';
            }
        }

        // 統計情報ビューをレンダリング
        function renderStatisticsView(stats) {
            const container = document.getElementById('statsViewContainer');

            let html = '';

            // サマリーセクション
            html += '<div class="stats-section">';
            html += '<div class="stats-section-title">Overview</div>';
            html += '<div class="stats-summary">';
            html += `
                <div class="stats-card">
                    <div class="stats-card-label">Total Datasets</div>
                    <div class="stats-card-value">${stats.total_datasets}</div>
                </div>
                <div class="stats-card">
                    <div class="stats-card-label">Total Windows</div>
                    <div class="stats-card-value">${stats.total_windows.toLocaleString()}</div>
                </div>
                <div class="stats-card">
                    <div class="stats-card-label">Total Users</div>
                    <div class="stats-card-value">${stats.total_users}</div>
                </div>
            `;
            html += '</div>';
            html += '</div>';

            // 各データセットの詳細
            stats.datasets.forEach((dataset, idx) => {
                html += '<div class="stats-section">';
                html += `<div class="stats-section-title">${dataset.name.toUpperCase()}</div>`;

                // データセットサマリー
                html += '<div class="stats-summary" style="margin-bottom: 20px;">';
                html += `
                    <div class="stats-card">
                        <div class="stats-card-label">Total Windows</div>
                        <div class="stats-card-value">${dataset.total_windows.toLocaleString()}</div>
                    </div>
                    <div class="stats-card">
                        <div class="stats-card-label">Users</div>
                        <div class="stats-card-value">${dataset.num_users}</div>
                    </div>
                    <div class="stats-card">
                        <div class="stats-card-label">Sensors</div>
                        <div class="stats-card-value">${dataset.num_sensors}</div>
                    </div>
                    <div class="stats-card">
                        <div class="stats-card-label">Activity Classes</div>
                        <div class="stats-card-value">${dataset.num_classes}</div>
                    </div>
                `;
                html += '</div>';

                // センサーリスト
                html += '<div style="margin-bottom: 20px;">';
                html += '<h3 style="font-size: 14px; color: #5f6368; margin-bottom: 8px;">Available Sensors:</h3>';
                html += '<div style="display: flex; flex-wrap: wrap; gap: 8px;">';
                dataset.sensors.forEach(sensor => {
                    html += `<span style="background: #e8f0fe; color: #1967d2; padding: 4px 12px; border-radius: 16px; font-size: 12px; font-weight: 500;">${sensor}</span>`;
                });
                html += '</div></div>';

                // 行動クラス別サマリー
                html += '<h3 style="font-size: 14px; color: #5f6368; margin-bottom: 12px;">Activity Class Summary:</h3>';
                html += '<table class="stats-table" style="margin-bottom: 20px;">';
                html += `
                    <thead>
                        <tr>
                            <th>Class ID</th>
                            <th>Activity Name</th>
                            <th>Total Windows</th>
                            <th>Sensors</th>
                            <th>Windows/Sensor</th>
                        </tr>
                    </thead>
                    <tbody>
                `;

                dataset.class_summary.forEach(classInfo => {
                    const windowsPerSensor = (classInfo.total_windows / classInfo.num_sensors).toFixed(1);
                    html += `
                        <tr>
                            <td><strong>${classInfo.class_id}</strong></td>
                            <td>${classInfo.name}</td>
                            <td><strong>${classInfo.total_windows.toLocaleString()}</strong></td>
                            <td>${classInfo.num_sensors} sensors</td>
                            <td style="color: #1a73e8; font-weight: 500;">${windowsPerSensor}</td>
                        </tr>
                    `;
                });

                html += '</tbody></table>';

                // 詳細テーブル（折りたたみ可能）
                html += `
                    <details style="margin-top: 16px;">
                        <summary style="cursor: pointer; padding: 8px; background: #f8f9fa; border-radius: 4px; font-weight: 500; color: #5f6368;">
                            Show detailed breakdown (${dataset.details.length} entries)
                        </summary>
                        <table class="stats-table" style="margin-top: 12px;">
                            <thead>
                                <tr>
                                    <th>User</th>
                                    <th>Sensor</th>
                                    <th>Activity Class</th>
                                    <th>Windows</th>
                                </tr>
                            </thead>
                            <tbody>
                `;

                dataset.details.forEach(detail => {
                    const activityName = detail.activity_name || `Class ${detail.class_id}`;
                    html += `
                        <tr>
                            <td>${detail.user}</td>
                            <td>${detail.sensor}</td>
                            <td>${activityName}</td>
                            <td><strong>${detail.count.toLocaleString()}</strong></td>
                        </tr>
                    `;
                });

                html += `
                            </tbody>
                        </table>
                    </details>
                `;

                html += '</div>';
            });

            container.innerHTML = html;
        }

        // ツリーデータを読み込み
        async function loadTree() {
            try {
                const response = await fetch('/api/tree');
                const tree = await response.json();
                renderTree(tree);
            } catch (error) {
                console.error('Failed to load tree:', error);
                document.getElementById('treeContainer').innerHTML = '<div class="loading">Error loading data</div>';
            }
        }

        // ツリーをレンダリング
        function renderTree(tree) {
            const container = document.getElementById('treeContainer');
            container.innerHTML = '';

            tree.forEach(dataset => {
                const datasetNode = createTreeNode(dataset, 'dataset');
                container.appendChild(datasetNode);
            });
        }

        // ツリーノードを作成
        function createTreeNode(node, type) {
            const div = document.createElement('div');

            if (type === 'dataset') {
                div.className = 'tree-node dataset collapsed';
                div.innerHTML = `${node.name}`;
                div.onclick = (e) => {
                    e.stopPropagation();
                    div.classList.toggle('collapsed');
                };

                const childrenDiv = document.createElement('div');
                childrenDiv.className = 'tree-children';
                node.children.forEach(user => {
                    childrenDiv.appendChild(createTreeNode(user, 'user'));
                });
                div.appendChild(childrenDiv);
            } else if (type === 'user') {
                div.className = 'tree-node user collapsed';
                div.innerHTML = `${node.name}`;
                div.onclick = (e) => {
                    e.stopPropagation();
                    div.classList.toggle('collapsed');
                };

                const childrenDiv = document.createElement('div');
                childrenDiv.className = 'tree-children';
                node.children.forEach(position => {
                    childrenDiv.appendChild(createTreeNode(position, 'position'));
                });
                div.appendChild(childrenDiv);
            } else if (type === 'position') {
                div.className = 'tree-node position collapsed';
                div.innerHTML = `${node.name}`;
                div.onclick = (e) => {
                    e.stopPropagation();
                    div.classList.toggle('collapsed');
                };

                const childrenDiv = document.createElement('div');
                childrenDiv.className = 'tree-children';
                node.children.forEach(modality => {
                    childrenDiv.appendChild(createTreeNode(modality, 'modality'));
                });
                div.appendChild(childrenDiv);
            } else if (type === 'modality') {
                div.className = 'tree-node modality collapsed';
                div.innerHTML = `${node.name}`;
                div.onclick = (e) => {
                    e.stopPropagation();
                    div.classList.toggle('collapsed');
                };

                const childrenDiv = document.createElement('div');
                childrenDiv.className = 'tree-children';
                node.children.forEach(cls => {
                    childrenDiv.appendChild(createTreeNode(cls, 'class'));
                });
                div.appendChild(childrenDiv);
            } else if (type === 'class') {
                div.className = 'tree-node class';
                div.innerHTML = `
                    <span>${node.name}</span>
                    <button class="add-btn" onclick="addPanel('${node.path}', event, ${node.class_id})">Add</button>
                `;
            }

            return div;
        }

        // パネルを追加
        async function addPanel(source, event, classId = null) {
            if (event) event.stopPropagation();

            const panelId = panelIdCounter++;
            const panel = {
                id: panelId,
                source: source,
                numSamples: 6,
                selectedClasses: classId !== null ? [classId] : null
            };

            panels.push(panel);
            await renderPanels();
        }

        // パネルを削除
        function removePanel(panelId) {
            panels = panels.filter(p => p.id !== panelId);
            renderPanels();
        }

        // 全パネルをクリア
        function clearAllPanels() {
            panels = [];
            renderPanels();
        }

        // パネルをレンダリング
        async function renderPanels() {
            const container = document.getElementById('panelsContainer');

            if (panels.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <h3>No panels added</h3>
                        <p>Select a sensor from the left sidebar to get started</p>
                    </div>
                `;
                return;
            }

            container.innerHTML = '';

            for (const panel of panels) {
                const panelDiv = await createPanelElement(panel);
                container.appendChild(panelDiv);
            }
        }

        // パネル要素を作成
        async function createPanelElement(panel) {
            const div = document.createElement('div');
            div.className = 'panel';
            div.id = `panel-${panel.id}`;

            // データを取得
            try {
                let url = `/api/panel_data?source=${encodeURIComponent(panel.source)}&num_samples=${panel.numSamples}&sampling=${globalSamplingMode}`;
                if (panel.selectedClasses) {
                    url += `&classes=${panel.selectedClasses.join(',')}`;
                }
                const response = await fetch(url);
                const data = await response.json();

                if (data.error) {
                    div.innerHTML = `<div class="loading">Error: ${data.error}</div>`;
                    return div;
                }

                const { metadata, samples } = data;

                // パネルヘッダー
                const headerHTML = `
                    <div class="panel-header">
                        <div class="panel-title">${metadata.dataset} / ${metadata.user} / ${metadata.sensor}</div>
                        <div class="panel-controls">
                            <button onclick="refreshPanel(${panel.id})">Refresh</button>
                            <button class="remove-btn" onclick="removePanel(${panel.id})">Remove</button>
                        </div>
                    </div>
                `;

                // メタデータ部分
                const metadataHTML = `
                    <div class="panel-metadata">
                        <div class="metadata-item">
                            <div class="metadata-label">Dataset</div>
                            <div class="metadata-value">${metadata.dataset}</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">User</div>
                            <div class="metadata-value">${metadata.user}</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">Sensor</div>
                            <div class="metadata-value">${metadata.sensor}</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">Channels</div>
                            <div class="metadata-value">${metadata.num_channels}</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">Window Size</div>
                            <div class="metadata-value">${metadata.window_size}</div>
                        </div>
                    </div>
                `;

                // プロット部分
                let plotsHTML = '<div class="panel-plots">';
                samples.forEach((sample, idx) => {
                    const plotId = `plot-${panel.id}-${idx}`;
                    plotsHTML += `
                        <div class="plot-item">
                            <div class="plot-header">
                                <strong>${sample.activity}</strong> (Sample #${sample.index})
                            </div>
                            <div id="${plotId}" style="width: 100%; height: 150px;"></div>
                        </div>
                    `;
                });
                plotsHTML += '</div>';

                div.innerHTML = headerHTML + '<div class="panel-content">' + metadataHTML + plotsHTML + '</div>';

                // Plotlyでグラフを描画
                setTimeout(() => {
                    samples.forEach((sample, idx) => {
                        const plotId = `plot-${panel.id}-${idx}`;
                        const isLastPlot = idx === samples.length - 1;
                        createPlot(plotId, sample.data, metadata.window_size, isLastPlot);
                    });
                }, 0);

            } catch (error) {
                console.error('Failed to load panel data:', error);
                div.innerHTML = '<div class="loading">Error loading data</div>';
            }

            return div;
        }

        // Plotlyグラフを作成
        function createPlot(plotId, data, windowSize, showLegend = false) {
            const numChannels = data.length;
            const xValues = Array.from({length: windowSize}, (_, i) => i);

            const traces = [];
            const colors = ['#ea4335', '#34a853', '#1a73e8'];
            const names = ['X-axis', 'Y-axis', 'Z-axis'];

            // 各軸
            for (let ch = 0; ch < Math.min(numChannels, 3); ch++) {
                traces.push({
                    x: xValues,
                    y: data[ch],
                    mode: 'lines',
                    name: names[ch],
                    line: { color: colors[ch], width: 1.5 }
                });
            }

            // Magnitude
            if (numChannels >= 3) {
                const magnitude = data[0].map((_, i) =>
                    Math.sqrt(data[0][i]**2 + data[1][i]**2 + data[2][i]**2)
                );
                traces.push({
                    x: xValues,
                    y: magnitude,
                    mode: 'lines',
                    name: 'Magnitude',
                    line: { color: '#9e9e9e', width: 2 }
                });
            }

            const layout = {
                autosize: true,
                margin: { t: 20, r: showLegend ? 80 : 20, b: 40, l: 50 },
                xaxis: { title: 'Time (samples)', titlefont: { size: 10 } },
                yaxis: { title: 'Value', titlefont: { size: 10 } },
                showlegend: showLegend,
                legend: {
                    x: 1.02,
                    y: 1,
                    xanchor: 'left',
                    font: { size: 9 },
                    orientation: 'v'
                },
                hovermode: 'closest',
                font: { size: 10 }
            };

            const config = {
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            };

            Plotly.newPlot(plotId, traces, layout, config);
        }


        // パネルをリフレッシュ
        async function refreshPanel(panelId) {
            await renderPanels();
        }

        // サンプリングモードを更新
        function updateSamplingMode() {
            globalSamplingMode = document.getElementById('samplingMode').value;
            // 全パネルを再描画
            renderPanels();
        }

        // 初期化
        loadTree();
    </script>
</body>
</html>
"""

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HAR Data Visualization</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            min-height: 100vh;
        }
        .header {
            background: white;
            border-bottom: 1px solid #e0e0e0;
            padding: 24px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 24px;
        }
        h1 {
            color: #202124;
            font-size: 28px;
            font-weight: 400;
            margin-bottom: 8px;
        }
        .subtitle {
            color: #5f6368;
            font-size: 14px;
        }
        .content {
            max-width: 1200px;
            margin: 32px auto;
            padding: 0 24px;
        }
        .dataset-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 24px;
        }
        .dataset-card {
            background: white;
            border-radius: 8px;
            padding: 24px;
            border: 1px solid #dadce0;
            transition: box-shadow 0.2s, transform 0.2s;
            cursor: pointer;
        }
        .dataset-card:hover {
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            transform: translateY(-2px);
        }
        .dataset-name {
            font-size: 20px;
            font-weight: 500;
            color: #1a73e8;
            margin-bottom: 16px;
        }
        .dataset-info {
            color: #5f6368;
            line-height: 1.6;
            font-size: 14px;
        }
        .dataset-info div {
            margin: 8px 0;
        }
        .badge {
            display: inline-block;
            background: #e8f0fe;
            color: #1967d2;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 12px;
            margin: 4px 4px 4px 0;
            font-weight: 500;
        }
        .no-datasets {
            background: white;
            padding: 48px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #dadce0;
        }
        .no-datasets h2 {
            color: #202124;
            font-size: 20px;
            font-weight: 400;
            margin-bottom: 16px;
        }
        .no-datasets p {
            color: #5f6368;
            margin-bottom: 16px;
        }
        .code-block {
            background: #f1f3f4;
            color: #202124;
            padding: 16px;
            border-radius: 4px;
            margin: 16px 0;
            font-family: 'Roboto Mono', monospace;
            text-align: left;
            white-space: pre;
            border: 1px solid #dadce0;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>HAR Data Visualization</h1>
            <div class="subtitle">Select a dataset to explore sensor data</div>
        </div>
    </div>

    <div class="content">
        {% if datasets %}
        <div class="dataset-grid">
            {% for dataset in datasets %}
            <div class="dataset-card" onclick="location.href='/dataset/{{ dataset.name }}'">
                <div class="dataset-name">{{ dataset.name.upper() }}</div>
                <div class="dataset-info">
                    <div><strong>Users:</strong> {{ dataset.num_users }}</div>
                    <div><strong>Activities:</strong> {{ dataset.num_activities }}</div>
                    <div><strong>Sensors:</strong> {{ dataset.num_sensors }}</div>
                    <div style="margin-top: 12px;">
                        {% for modality in dataset.modalities %}
                        <span class="badge">{{ modality }}</span>
                        {% endfor %}
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        {% else %}
        <div class="no-datasets">
            <h2>No datasets found</h2>
            <p>Please preprocess a dataset first:</p>
            <div class="code-block">python preprocess.py --dataset dsads --download</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

DATASET_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ dataset_name.upper() }} - Visualization</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            min-height: 100vh;
        }
        .header {
            background: white;
            border-bottom: 1px solid #e0e0e0;
            padding: 24px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 24px;
        }
        .back-link {
            display: inline-block;
            color: #1a73e8;
            text-decoration: none;
            margin-bottom: 12px;
            font-size: 14px;
        }
        .back-link:hover {
            text-decoration: underline;
        }
        h1 {
            font-size: 28px;
            font-weight: 400;
            color: #202124;
            margin-bottom: 16px;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            margin-top: 16px;
        }
        .info-item {
            background: #f1f3f4;
            padding: 12px;
            border-radius: 4px;
            font-size: 14px;
            color: #5f6368;
        }
        .info-item strong {
            color: #202124;
        }
        .content {
            max-width: 1200px;
            margin: 24px auto;
            padding: 0 24px;
        }
        .user-section {
            background: white;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid #dadce0;
        }
        .user-title {
            font-size: 18px;
            font-weight: 500;
            color: #202124;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid #e0e0e0;
        }
        .sensor-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 12px;
        }
        .sensor-btn {
            background: white;
            color: #1a73e8;
            border: 1px solid #dadce0;
            padding: 12px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: background 0.2s, border-color 0.2s, box-shadow 0.2s;
            font-family: 'Roboto', sans-serif;
        }
        .sensor-btn:hover {
            background: #f8f9fa;
            border-color: #1a73e8;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        .sensor-btn:active {
            background: #e8f0fe;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <a href="/" class="back-link">← Back to datasets</a>
            <h1>{{ dataset_name.upper() }}</h1>
            <div class="info-grid">
                <div class="info-item">
                    <strong>Users:</strong> {{ metadata.users|length }}
                </div>
                <div class="info-item">
                    <strong>Activities:</strong> {{ metadata.num_activities }}
                </div>
                <div class="info-item">
                    <strong>Window Size:</strong> {{ metadata.window_size }}
                </div>
                <div class="info-item">
                    <strong>Stride:</strong> {{ metadata.stride }}
                </div>
            </div>
        </div>
    </div>

    <div class="content">
        {% for user_id, user_data in metadata.users.items() %}
        <div class="user-section">
            <div class="user-title">{{ user_id }}</div>
            <div class="sensor-grid">
                {% for sensor_name in user_data.sensor_modalities.keys()|sort %}
                <button class="sensor-btn"
                        onclick="location.href='/visualize/{{ dataset_name }}/{{ user_id }}/{{ sensor_name }}'">
                    {{ sensor_name }}
                </button>
                {% endfor %}
            </div>
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""


def get_available_datasets():
    """利用可能なデータセットのリストを取得"""
    if not DATA_DIR.exists():
        return []

    datasets = []
    for dataset_path in DATA_DIR.iterdir():
        if dataset_path.is_dir():
            metadata_path = dataset_path / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                datasets.append({
                    'name': dataset_path.name,
                    'num_users': len(metadata.get('users', {})),
                    'num_activities': metadata.get('num_activities', 'N/A'),
                    'num_sensors': metadata.get('num_sensors', 'N/A'),
                    'modalities': metadata.get('modalities', [])
                })

    return datasets


def load_metadata(dataset_name):
    """データセットのメタデータを読み込む"""
    metadata_path = DATA_DIR / dataset_name / 'metadata.json'
    if not metadata_path.exists():
        return None

    with open(metadata_path, 'r') as f:
        return json.load(f)


def load_sensor_data(dataset_name, user_id, sensor_name):
    """センサーデータを読み込む

    データ構造: USER00001/LeftArm/ACC/X.npy
    sensor_name: "LeftArm/ACC"
    """
    parts = sensor_name.split('/')
    data_path = DATA_DIR / dataset_name / user_id / parts[0] / parts[1]

    X_path = data_path / 'X.npy'
    Y_path = data_path / 'Y.npy'

    if not X_path.exists() or not Y_path.exists():
        return None, None

    X = np.load(X_path)
    Y = np.load(Y_path)
    return X, Y


def create_visualization(X, Y, dataset_name, user_id, sensor_name,
                        selected_classes=None, grid_layout='2x2'):
    """可視化を生成 - 各クラスから個別のウィンドウをサンプリング

    Args:
        X: センサーデータ (samples, channels, window_size)
        Y: ラベル
        dataset_name: データセット名
        user_id: ユーザーID
        sensor_name: センサー名
        selected_classes: 表示する行動クラスのリスト（Noneの場合は全て）
        grid_layout: グリッドレイアウト ('1x1', '2x2', '3x3', etc.)
    """
    num_channels = X.shape[1]
    window_size = X.shape[2]

    # 軸の色を固定（X軸=赤、Y軸=緑、Z軸=青）
    AXIS_COLORS = ['#ea4335', '#34a853', '#1a73e8']  # Red, Green, Blue
    AXIS_NAMES = ['X-axis', 'Y-axis', 'Z-axis']

    # クラスフィルタリング
    if selected_classes is not None:
        mask = np.isin(Y, selected_classes)
        X = X[mask]
        Y = Y[mask]

    if len(Y) == 0:
        # データがない場合はエラーメッセージを返す
        return "<html><body><h2>No data available for the selected classes</h2></body></html>"

    # グリッドレイアウトのパース
    rows, cols = map(int, grid_layout.split('x'))
    total_subplots = rows * cols

    # 各クラスから均等にサンプリング（グリッドサイズに合わせて）
    unique_labels = np.unique(Y)
    samples_per_class = max(1, total_subplots // len(unique_labels))

    # サンプルを収集
    sampled_data = []
    for label in unique_labels:
        indices = np.where(Y == label)[0]
        n_samples = min(samples_per_class, len(indices))
        selected = np.random.choice(indices, n_samples, replace=False)
        for idx in selected:
            sampled_data.append({
                'index': idx,
                'label': label,
                'data': X[idx]
            })

    # クラスでソート
    sampled_data.sort(key=lambda x: x['label'])

    # グリッドサイズに合わせて調整
    sampled_data = sampled_data[:total_subplots]

    # サブプロット作成
    subplot_titles = []
    for s in sampled_data:
        label = int(s['label'])
        activity = get_activity_name(dataset_name, label)
        subplot_titles.append(f'Sample {s["index"]}: {activity}')

    # 余白を調整（グリッドが大きいほど余白を大きく）
    vertical_spacing = max(0.12 / rows, 0.03)
    horizontal_spacing = max(0.10 / cols, 0.03)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing
    )

    # 各サンプルをプロット
    for plot_idx, sample in enumerate(sampled_data):
        row = plot_idx // cols + 1
        col = plot_idx % cols + 1

        label = int(sample['label'])
        activity_name = get_activity_name(dataset_name, label)
        data = sample['data']

        for ch in range(min(num_channels, 3)):  # 最大3軸まで表示
            x_vals = np.arange(window_size)
            y_vals = data[ch, :]

            # 最初のサブプロットでのみ凡例を表示
            showlegend = (plot_idx == 0)

            axis_name = AXIS_NAMES[ch] if ch < len(AXIS_NAMES) else f'Channel {ch}'
            axis_color = AXIS_COLORS[ch] if ch < len(AXIS_COLORS) else '#5f6368'

            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines',
                    name=axis_name,
                    line=dict(color=axis_color, width=1.5),
                    legendgroup=f'axis_{ch}',
                    showlegend=showlegend,
                    hovertemplate=f'{axis_name}<br>Time: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>'
                ),
                row=row, col=col
            )

        # ノルム（大きさ）を計算して追加
        if num_channels >= 3:
            norm = np.sqrt(data[0, :]**2 + data[1, :]**2 + data[2, :]**2)
            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=norm,
                    mode='lines',
                    name='Magnitude',
                    line=dict(color='#202124', width=2, dash='solid'),
                    legendgroup='magnitude',
                    showlegend=(plot_idx == 0),
                    hovertemplate='Magnitude<br>Time: %{x}<br>Value: %{y:.3f}<extra></extra>'
                ),
                row=row, col=col
            )

    # レイアウト（余白を考慮して高さを調整）
    subplot_height = 350
    total_height = subplot_height * rows + 100  # タイトルと余白を追加

    fig.update_layout(
        title=dict(
            text=f'{dataset_name.upper()} - {user_id} - {sensor_name}',
            font=dict(size=20, color='#202124')
        ),
        height=total_height,
        hovermode='closest',
        showlegend=True,
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Roboto, sans-serif', color='#5f6368'),
        margin=dict(t=100, b=50)  # 上下のマージンを確保
    )

    # 軸ラベル
    for i in range(len(sampled_data)):
        row = i // cols + 1
        col = i % cols + 1
        fig.update_xaxes(title_text="Time (samples)", row=row, col=col,
                        gridcolor='#e0e0e0', linecolor='#dadce0')
        fig.update_yaxes(title_text="Value", row=row, col=col,
                        gridcolor='#e0e0e0', linecolor='#dadce0')

    return fig.to_html(full_html=True, include_plotlyjs='cdn')


@app.route('/')
def index():
    """トップページ：新しいUI"""
    return render_template_string(NEW_UI_TEMPLATE)


@app.route('/api/tree')
def api_tree():
    """ツリービュー用のデータセット階層構造を返す

    階層: Dataset → User → Position (LeftArm) → Modality (ACC) → Class
    """
    tree = []

    if not DATA_DIR.exists():
        return jsonify(tree)

    for dataset_path in sorted(DATA_DIR.iterdir()):
        if not dataset_path.is_dir():
            continue

        dataset_node = {
            'name': dataset_path.name,
            'type': 'dataset',
            'children': []
        }

        # ユーザーディレクトリを探索
        for user_path in sorted(dataset_path.iterdir()):
            if not user_path.is_dir() or user_path.name == 'metadata.json':
                continue

            user_node = {
                'name': user_path.name,
                'type': 'user',
                'children': []
            }

            # 位置ディレクトリ（LeftArm, RightArm など）を探索
            for position_path in sorted(user_path.iterdir()):
                if not position_path.is_dir():
                    continue

                position_node = {
                    'name': position_path.name,
                    'type': 'position',
                    'children': []
                }

                # モダリティディレクトリ（ACC, GYRO, MAG）を探索
                for modality_path in sorted(position_path.iterdir()):
                    if not modality_path.is_dir() or not (modality_path / 'X.npy').exists():
                        continue

                    sensor_name = f'{position_path.name}/{modality_path.name}'
                    X, Y = load_sensor_data(dataset_path.name, user_path.name, sensor_name)

                    modality_node = {
                        'name': modality_path.name,
                        'type': 'modality',
                        'children': []
                    }

                    if X is not None and Y is not None:
                        # 利用可能なクラスを取得
                        unique_classes = sorted(np.unique(Y).astype(int).tolist())

                        for cls in unique_classes:
                            activity_name = get_activity_name(dataset_path.name, cls)
                            class_node = {
                                'name': f'{activity_name} (Class {cls})',
                                'type': 'class',
                                'path': f'{dataset_path.name}/{user_path.name}/{sensor_name}',
                                'class_id': cls
                            }
                            modality_node['children'].append(class_node)

                    if modality_node['children']:
                        position_node['children'].append(modality_node)

                if position_node['children']:
                    user_node['children'].append(position_node)

            if user_node['children']:
                dataset_node['children'].append(user_node)

        if dataset_node['children']:
            tree.append(dataset_node)

    return jsonify(tree)


@app.route('/api/statistics')
def api_statistics():
    """全データセットの統計情報を返す"""
    if not DATA_DIR.exists():
        return jsonify({'error': 'Data directory not found'}), 404

    stats = {
        'total_datasets': 0,
        'total_windows': 0,
        'total_users': 0,
        'datasets': []
    }

    all_users = set()

    for dataset_path in sorted(DATA_DIR.iterdir()):
        if not dataset_path.is_dir():
            continue

        dataset_stats = {
            'name': dataset_path.name,
            'total_windows': 0,
            'num_users': 0,
            'num_sensors': 0,
            'num_classes': 0,
            'sensors': set(),
            'classes': {},  # class_id -> {name, total_windows, sensors: {sensor_name: count}}
            'details': []
        }

        dataset_users = set()

        # ユーザーディレクトリを探索
        for user_path in sorted(dataset_path.iterdir()):
            if not user_path.is_dir() or user_path.name == 'metadata.json':
                continue

            all_users.add(user_path.name)
            dataset_users.add(user_path.name)

            # 位置ディレクトリを探索
            for position_path in sorted(user_path.iterdir()):
                if not position_path.is_dir():
                    continue

                # モダリティディレクトリを探索
                for modality_path in sorted(position_path.iterdir()):
                    if not modality_path.is_dir():
                        continue

                    Y_path = modality_path / 'Y.npy'
                    if not Y_path.exists():
                        continue

                    sensor_name = f'{position_path.name}/{modality_path.name}'
                    dataset_stats['sensors'].add(sensor_name)

                    try:
                        Y = np.load(Y_path)
                        unique_classes = np.unique(Y)

                        # クラスごとにカウント
                        for cls in unique_classes:
                            count = int(np.sum(Y == cls))
                            activity_name = get_activity_name(dataset_path.name, int(cls))
                            class_id = int(cls)

                            # クラス別集計
                            if class_id not in dataset_stats['classes']:
                                dataset_stats['classes'][class_id] = {
                                    'name': activity_name,
                                    'total_windows': 0,
                                    'sensors': {}
                                }

                            dataset_stats['classes'][class_id]['total_windows'] += count
                            dataset_stats['classes'][class_id]['sensors'][sensor_name] = \
                                dataset_stats['classes'][class_id]['sensors'].get(sensor_name, 0) + count

                            dataset_stats['details'].append({
                                'user': user_path.name,
                                'position': position_path.name,
                                'modality': modality_path.name,
                                'sensor': sensor_name,
                                'class_id': class_id,
                                'activity_name': activity_name,
                                'count': count
                            })

                            dataset_stats['total_windows'] += count

                    except Exception as e:
                        print(f"Error loading {Y_path}: {e}")
                        continue

        if dataset_stats['details']:
            dataset_stats['num_users'] = len(dataset_users)
            dataset_stats['num_sensors'] = len(dataset_stats['sensors'])
            dataset_stats['num_classes'] = len(dataset_stats['classes'])

            # セットをリストに変換（JSON化のため）
            dataset_stats['sensors'] = sorted(list(dataset_stats['sensors']))

            # クラス情報を整形
            dataset_stats['class_summary'] = []
            for class_id in sorted(dataset_stats['classes'].keys()):
                class_info = dataset_stats['classes'][class_id]
                dataset_stats['class_summary'].append({
                    'class_id': class_id,
                    'name': class_info['name'] or f'Class {class_id}',
                    'total_windows': class_info['total_windows'],
                    'num_sensors': len(class_info['sensors']),
                    'sensors': class_info['sensors']
                })

            del dataset_stats['classes']  # 元の辞書形式は削除

            stats['datasets'].append(dataset_stats)
            stats['total_windows'] += dataset_stats['total_windows']
            stats['total_datasets'] += 1

    stats['total_users'] = len(all_users)

    return jsonify(stats)


@app.route('/api/panel_data')
def api_panel_data():
    """パネル用のデータを返す"""
    source = request.args.get('source', '')  # dataset/user/position/modality
    num_samples = int(request.args.get('num_samples', 5))
    selected_classes_param = request.args.get('classes', None)
    sampling_mode = request.args.get('sampling', 'random')  # 'random' or 'sequential'

    parts = source.split('/')
    if len(parts) != 4:
        return jsonify({'error': f'Invalid source format: expected 4 parts, got {len(parts)}'}), 400

    dataset_name, user_id, position, modality = parts
    sensor_name = f'{position}/{modality}'

    # データ読み込み
    X, Y = load_sensor_data(dataset_name, user_id, sensor_name)
    if X is None or Y is None:
        return jsonify({'error': 'Data not found'}), 404

    # クラスフィルタ
    selected_classes = None
    if selected_classes_param:
        try:
            selected_classes = [int(c) for c in selected_classes_param.split(',')]
            mask = np.isin(Y, selected_classes)
            X = X[mask]
            Y = Y[mask]
        except ValueError:
            pass

    if len(Y) == 0:
        return jsonify({'error': 'No data available for selected classes'}), 404

    # 各クラスから均等にサンプリング
    unique_labels = np.unique(Y)
    samples_per_class = max(1, num_samples // len(unique_labels))

    sampled_data = []
    for label in unique_labels:
        indices = np.where(Y == label)[0]
        n_samples = min(samples_per_class, len(indices))

        if sampling_mode == 'sequential':
            # 先頭から順番に取得
            selected = indices[:n_samples]
        else:
            # ランダムに取得
            selected = np.random.choice(indices, n_samples, replace=False)

        for idx in selected:
            sampled_data.append({
                'index': int(idx),
                'label': int(label),
                'activity': get_activity_name(dataset_name, int(label)),
                'data': X[idx].tolist()
            })

    # クラスでソート
    sampled_data.sort(key=lambda x: x['label'])
    sampled_data = sampled_data[:num_samples]

    # メタデータ
    metadata = {
        'dataset': dataset_name,
        'user': user_id,
        'sensor': sensor_name,
        'num_channels': X.shape[1],
        'window_size': X.shape[2],
        'available_classes': [int(l) for l in np.unique(Y)]
    }

    return jsonify({
        'metadata': metadata,
        'samples': sampled_data
    })


@app.route('/old')
def old_index():
    """旧トップページ：データセット一覧"""
    datasets = get_available_datasets()
    return render_template_string(INDEX_TEMPLATE, datasets=datasets)


@app.route('/dataset/<dataset_name>')
def dataset_detail(dataset_name):
    """データセット詳細：ユーザー×センサーグリッド"""
    metadata = load_metadata(dataset_name)
    if metadata is None:
        return f"Dataset '{dataset_name}' not found", 404

    return render_template_string(DATASET_TEMPLATE,
                                  dataset_name=dataset_name,
                                  metadata=metadata)


def create_comparison_visualization(all_data, selected_classes=None, grid_layout='2x2'):
    """複数ソースの比較可視化を生成"""
    rows, cols = map(int, grid_layout.split('x'))

    # 軸の色を固定
    AXIS_COLORS = ['#ea4335', '#34a853', '#1a73e8']
    AXIS_NAMES = ['X-axis', 'Y-axis', 'Z-axis']

    # サブプロットタイトル
    subplot_titles = []
    filtered_data = []

    for data_source in all_data[:rows * cols]:
        X, Y = data_source['X'], data_source['Y']

        # クラスフィルタリング
        if selected_classes is not None:
            mask = np.isin(Y, selected_classes)
            X = X[mask]
            Y = Y[mask]

        if len(Y) > 0:
            # ランダムにサンプルを1つ選択
            idx = np.random.choice(len(Y))
            filtered_data.append({
                'X': X[idx],
                'Y': Y[idx],
                'idx': idx,
                'dataset': data_source['dataset'],
                'user': data_source['user'],
                'sensor': data_source['sensor']
            })

            activity_name = get_activity_name(dataset_name, int(Y[idx]))
            title = f"{data_source['dataset']}/{data_source['user']}<br>{data_source['sensor']}<br>{activity_name}"
            subplot_titles.append(title)

    if not filtered_data:
        return "<html><body><h2>No data available for the selected classes</h2></body></html>"

    # 余白を調整
    vertical_spacing = max(0.15 / rows, 0.05)
    horizontal_spacing = max(0.10 / cols, 0.03)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing
    )

    # 各ソースをプロット
    for plot_idx, sample in enumerate(filtered_data):
        row = plot_idx // cols + 1
        col = plot_idx % cols + 1

        data = sample['X']
        num_channels = data.shape[0]
        window_size = data.shape[1]

        # 各軸をプロット
        for ch in range(min(num_channels, 3)):
            x_vals = np.arange(window_size)
            y_vals = data[ch, :]

            showlegend = (plot_idx == 0)
            axis_name = AXIS_NAMES[ch] if ch < len(AXIS_NAMES) else f'Channel {ch}'
            axis_color = AXIS_COLORS[ch] if ch < len(AXIS_COLORS) else '#5f6368'

            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines',
                    name=axis_name,
                    line=dict(color=axis_color, width=1.5),
                    legendgroup=f'axis_{ch}',
                    showlegend=showlegend,
                    hovertemplate=f'{axis_name}<br>Time: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>'
                ),
                row=row, col=col
            )

        # ノルム
        if num_channels >= 3:
            norm = np.sqrt(data[0, :]**2 + data[1, :]**2 + data[2, :]**2)
            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=norm,
                    mode='lines',
                    name='Magnitude',
                    line=dict(color='#202124', width=2),
                    legendgroup='magnitude',
                    showlegend=(plot_idx == 0),
                    hovertemplate='Magnitude<br>Time: %{x}<br>Value: %{y:.3f}<extra></extra>'
                ),
                row=row, col=col
            )

    # レイアウト
    subplot_height = 350
    total_height = subplot_height * rows + 150

    fig.update_layout(
        title=dict(
            text='Sensor Data Comparison',
            font=dict(size=20, color='#202124')
        ),
        height=total_height,
        hovermode='closest',
        showlegend=True,
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Roboto, sans-serif', color='#5f6368'),
        margin=dict(t=120, b=50)
    )

    # 軸ラベル
    for i in range(len(filtered_data)):
        row = i // cols + 1
        col = i % cols + 1
        fig.update_xaxes(title_text="Time (samples)", row=row, col=col,
                        gridcolor='#e0e0e0', linecolor='#dadce0')
        fig.update_yaxes(title_text="Value", row=row, col=col,
                        gridcolor='#e0e0e0', linecolor='#dadce0')

    return fig.to_html(full_html=True, include_plotlyjs='cdn')


def generate_comparison_page(plot_html, all_data, sources, selected_classes, grid_layout):
    """比較ページのHTMLを生成"""
    # 全ソースから利用可能なクラスを取得
    all_classes = set()
    for data_source in all_data:
        all_classes.update(np.unique(data_source['Y']).astype(int).tolist())

    unique_classes = sorted(all_classes)

    # クラス選択のチェックボックスHTML
    class_checkboxes = []
    for cls in unique_classes:
        activity_name = get_activity_name(dataset_name, cls)
        checked = 'checked' if (selected_classes is None or cls in selected_classes) else ''
        class_checkboxes.append(
            f'<label class="checkbox-label">'
            f'<input type="checkbox" name="class" value="{cls}" {checked}> '
            f'{activity_name}'
            f'</label>'
        )

    class_checkboxes_html = '\n'.join(class_checkboxes)

    # グリッドレイアウトの選択肢
    grid_options = ['1x1', '2x2', '3x3', '4x4', '2x3', '3x2', '4x3', '5x4']
    grid_select_html = []
    for option in grid_options:
        selected = 'selected' if option == grid_layout else ''
        grid_select_html.append(f'<option value="{option}" {selected}>{option}</option>')

    grid_select_html = '\n'.join(grid_select_html)

    # ソースリスト
    sources_list_html = []
    for i, source in enumerate(sources, 1):
        sources_list_html.append(
            f'<div class="source-item">{i}. {source["dataset"]}/{source["user"]}/{source["sensor"]}</div>'
        )
    sources_list_html = '\n'.join(sources_list_html)

    # ソースパラメータを再構築
    sources_param = ','.join([f'{s["dataset"]}/{s["user"]}/{s["sensor"]}' for s in sources])

    return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sensor Data Comparison</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            min-height: 100vh;
        }}
        .header {{
            background: white;
            border-bottom: 1px solid #e0e0e0;
            padding: 16px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            position: sticky;
            top: 0;
            z-index: 1000;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 24px;
        }}
        .back-link {{
            display: inline-block;
            color: #1a73e8;
            text-decoration: none;
            margin-bottom: 8px;
            font-size: 14px;
        }}
        .back-link:hover {{ text-decoration: underline; }}
        h1 {{
            font-size: 20px;
            font-weight: 500;
            color: #202124;
            margin-bottom: 12px;
        }}
        .controls {{
            background: white;
            border: 1px solid #dadce0;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 24px;
            max-width: 1400px;
            margin-left: auto;
            margin-right: auto;
        }}
        .control-section {{
            margin-bottom: 16px;
        }}
        .control-section:last-child {{
            margin-bottom: 0;
        }}
        .control-label {{
            font-weight: 500;
            color: #202124;
            margin-bottom: 8px;
            display: block;
            font-size: 14px;
        }}
        .class-filters {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 8px;
        }}
        .checkbox-label {{
            display: flex;
            align-items: center;
            padding: 6px 8px;
            background: #f8f9fa;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            color: #5f6368;
            transition: background 0.2s;
        }}
        .checkbox-label:hover {{
            background: #e8f0fe;
        }}
        .checkbox-label input {{
            margin-right: 8px;
            cursor: pointer;
        }}
        .control-row {{
            display: flex;
            gap: 16px;
            align-items: end;
            flex-wrap: wrap;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
        }}
        select {{
            padding: 8px 12px;
            border: 1px solid #dadce0;
            border-radius: 4px;
            font-size: 14px;
            font-family: inherit;
            background: white;
            min-width: 120px;
        }}
        .btn {{
            background: #1a73e8;
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: background 0.2s;
        }}
        .btn:hover {{
            background: #1557b0;
        }}
        .btn-secondary {{
            background: white;
            color: #5f6368;
            border: 1px solid #dadce0;
        }}
        .btn-secondary:hover {{
            background: #f8f9fa;
        }}
        .plot-container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 24px 24px 24px;
        }}
        .legend-info {{
            background: #e8f0fe;
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 12px;
            font-size: 13px;
            color: #1967d2;
        }}
        .legend-info strong {{
            color: #1557b0;
        }}
        .axis-legend {{
            display: flex;
            gap: 16px;
            margin-top: 8px;
        }}
        .axis-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .axis-color {{
            width: 20px;
            height: 3px;
            border-radius: 2px;
        }}
        .sources-list {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 4px;
            margin-top: 12px;
        }}
        .source-item {{
            padding: 4px 0;
            color: #5f6368;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <a href="/" class="back-link">← Back to datasets</a>
            <h1>Sensor Data Comparison</h1>
        </div>
    </div>

    <div class="controls">
        <form id="filterForm" method="get">
            <input type="hidden" name="sources" value="{sources_param}">

            <div class="control-section">
                <label class="control-label">比較中のソース:</label>
                <div class="sources-list">
                    {sources_list_html}
                </div>
            </div>

            <div class="control-section">
                <label class="control-label">行動クラスフィルタ:</label>
                <div class="class-filters">
                    {class_checkboxes_html}
                </div>
            </div>

            <div class="control-section">
                <div class="control-row">
                    <div class="control-group">
                        <label class="control-label" for="grid">グリッドレイアウト:</label>
                        <select id="grid" name="grid">
                            {grid_select_html}
                        </select>
                    </div>
                    <button type="submit" class="btn">更新</button>
                    <button type="button" class="btn btn-secondary" onclick="selectAllClasses()">全選択</button>
                    <button type="button" class="btn btn-secondary" onclick="deselectAllClasses()">全解除</button>
                </div>
            </div>
        </form>
    </div>

    <div class="plot-container">
        <div class="legend-info">
            <strong>軸の色:</strong>
            <div class="axis-legend">
                <div class="axis-item">
                    <div class="axis-color" style="background: #ea4335;"></div>
                    <span>X軸</span>
                </div>
                <div class="axis-item">
                    <div class="axis-color" style="background: #34a853;"></div>
                    <span>Y軸</span>
                </div>
                <div class="axis-item">
                    <div class="axis-color" style="background: #1a73e8;"></div>
                    <span>Z軸</span>
                </div>
                <div class="axis-item">
                    <div class="axis-color" style="background: #202124;"></div>
                    <span>Magnitude (√(x²+y²+z²))</span>
                </div>
            </div>
        </div>
        {plot_html}
    </div>

    <script>
        function selectAllClasses() {{
            document.querySelectorAll('input[name="class"]').forEach(cb => cb.checked = true);
        }}

        function deselectAllClasses() {{
            document.querySelectorAll('input[name="class"]').forEach(cb => cb.checked = false);
        }}

        document.getElementById('filterForm').addEventListener('submit', function(e) {{
            e.preventDefault();
            const checkedClasses = Array.from(document.querySelectorAll('input[name="class"]:checked'))
                                        .map(cb => cb.value)
                                        .join(',');
            const grid = document.getElementById('grid').value;
            const sources = document.querySelector('input[name="sources"]').value;

            let url = '/compare?sources=' + encodeURIComponent(sources) + '&grid=' + grid;
            if (checkedClasses) {{
                url += '&classes=' + checkedClasses;
            }}

            window.location.href = url;
        }});
    </script>
</body>
</html>
"""


def generate_visualization_page(plot_html, dataset_name, user_id, sensor_name,
                               unique_classes, selected_classes, grid_layout):
    """可視化ページのHTMLを生成"""
    # クラス選択のチェックボックスHTML
    class_checkboxes = []
    for cls in unique_classes:
        activity_name = get_activity_name(dataset_name, cls)
        checked = 'checked' if (selected_classes is None or cls in selected_classes) else ''
        class_checkboxes.append(
            f'<label class="checkbox-label">'
            f'<input type="checkbox" name="class" value="{cls}" {checked}> '
            f'{activity_name}'
            f'</label>'
        )

    class_checkboxes_html = '\n'.join(class_checkboxes)

    # グリッドレイアウトの選択肢
    grid_options = ['1x1', '2x2', '3x3', '4x4', '2x3', '3x2', '4x3', '5x4']
    grid_select_html = []
    for option in grid_options:
        selected = 'selected' if option == grid_layout else ''
        grid_select_html.append(f'<option value="{option}" {selected}>{option}</option>')

    grid_select_html = '\n'.join(grid_select_html)

    return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dataset_name.upper()} - {user_id} - {sensor_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            min-height: 100vh;
        }}
        .header {{
            background: white;
            border-bottom: 1px solid #e0e0e0;
            padding: 16px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            position: sticky;
            top: 0;
            z-index: 1000;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 24px;
        }}
        .back-link {{
            display: inline-block;
            color: #1a73e8;
            text-decoration: none;
            margin-bottom: 8px;
            font-size: 14px;
        }}
        .back-link:hover {{ text-decoration: underline; }}
        h1 {{
            font-size: 20px;
            font-weight: 500;
            color: #202124;
            margin-bottom: 12px;
        }}
        .controls {{
            background: white;
            border: 1px solid #dadce0;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 24px;
            max-width: 1400px;
            margin-left: auto;
            margin-right: auto;
        }}
        .control-section {{
            margin-bottom: 16px;
        }}
        .control-section:last-child {{
            margin-bottom: 0;
        }}
        .control-label {{
            font-weight: 500;
            color: #202124;
            margin-bottom: 8px;
            display: block;
            font-size: 14px;
        }}
        .class-filters {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 8px;
        }}
        .checkbox-label {{
            display: flex;
            align-items: center;
            padding: 6px 8px;
            background: #f8f9fa;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            color: #5f6368;
            transition: background 0.2s;
        }}
        .checkbox-label:hover {{
            background: #e8f0fe;
        }}
        .checkbox-label input {{
            margin-right: 8px;
            cursor: pointer;
        }}
        .control-row {{
            display: flex;
            gap: 16px;
            align-items: end;
            flex-wrap: wrap;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
        }}
        select, input[type="number"] {{
            padding: 8px 12px;
            border: 1px solid #dadce0;
            border-radius: 4px;
            font-size: 14px;
            font-family: inherit;
            background: white;
            min-width: 120px;
        }}
        .btn {{
            background: #1a73e8;
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: background 0.2s;
        }}
        .btn:hover {{
            background: #1557b0;
        }}
        .btn-secondary {{
            background: white;
            color: #5f6368;
            border: 1px solid #dadce0;
        }}
        .btn-secondary:hover {{
            background: #f8f9fa;
        }}
        .plot-container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 24px 24px 24px;
        }}
        .legend-info {{
            background: #e8f0fe;
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 12px;
            font-size: 13px;
            color: #1967d2;
        }}
        .legend-info strong {{
            color: #1557b0;
        }}
        .axis-legend {{
            display: flex;
            gap: 16px;
            margin-top: 8px;
        }}
        .axis-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .axis-color {{
            width: 20px;
            height: 3px;
            border-radius: 2px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <a href="/dataset/{dataset_name}" class="back-link">← Back to {dataset_name.upper()}</a>
            <h1>{dataset_name.upper()} - {user_id} - {sensor_name}</h1>
        </div>
    </div>

    <div class="controls">
        <form id="filterForm" method="get">
            <div class="control-section">
                <label class="control-label">行動クラスフィルタ:</label>
                <div class="class-filters">
                    {class_checkboxes_html}
                </div>
            </div>

            <div class="control-section">
                <div class="control-row">
                    <div class="control-group">
                        <label class="control-label" for="grid">グリッドレイアウト:</label>
                        <select id="grid" name="grid">
                            {grid_select_html}
                        </select>
                    </div>
                    <button type="submit" class="btn">更新</button>
                    <button type="button" class="btn btn-secondary" onclick="selectAllClasses()">全選択</button>
                    <button type="button" class="btn btn-secondary" onclick="deselectAllClasses()">全解除</button>
                </div>
            </div>
        </form>
    </div>

    <div class="plot-container">
        <div class="legend-info">
            <strong>軸の色:</strong>
            <div class="axis-legend">
                <div class="axis-item">
                    <div class="axis-color" style="background: #ea4335;"></div>
                    <span>X軸</span>
                </div>
                <div class="axis-item">
                    <div class="axis-color" style="background: #34a853;"></div>
                    <span>Y軸</span>
                </div>
                <div class="axis-item">
                    <div class="axis-color" style="background: #1a73e8;"></div>
                    <span>Z軸</span>
                </div>
                <div class="axis-item">
                    <div class="axis-color" style="background: #202124;"></div>
                    <span>Magnitude (√(x²+y²+z²))</span>
                </div>
            </div>
        </div>
        {plot_html}
    </div>

    <script>
        function selectAllClasses() {{
            document.querySelectorAll('input[name="class"]').forEach(cb => cb.checked = true);
        }}

        function deselectAllClasses() {{
            document.querySelectorAll('input[name="class"]').forEach(cb => cb.checked = false);
        }}

        // フォーム送信時に選択されたクラスをカンマ区切りで送信
        document.getElementById('filterForm').addEventListener('submit', function(e) {{
            e.preventDefault();
            const checkedClasses = Array.from(document.querySelectorAll('input[name="class"]:checked'))
                                        .map(cb => cb.value)
                                        .join(',');
            const grid = document.getElementById('grid').value;

            let url = window.location.pathname + '?grid=' + grid;
            if (checkedClasses) {{
                url += '&classes=' + checkedClasses;
            }}

            window.location.href = url;
        }});
    </script>
</body>
</html>
"""


@app.route('/visualize/<dataset_name>/<user_id>/<sensor_name>')
def visualize(dataset_name, user_id, sensor_name):
    """可視化ページ"""
    X, Y = load_sensor_data(dataset_name, user_id, sensor_name)

    if X is None or Y is None:
        return f"Data not found for {dataset_name}/{user_id}/{sensor_name}", 404

    # クエリパラメータから設定を取得
    grid_layout = request.args.get('grid', '2x2')
    selected_classes_param = request.args.get('classes', None)

    # クラスフィルタのパース
    selected_classes = None
    if selected_classes_param:
        try:
            selected_classes = [int(c) for c in selected_classes_param.split(',')]
        except ValueError:
            pass

    # 可視化を生成
    plot_html = create_visualization(
        X, Y, dataset_name, user_id, sensor_name,
        selected_classes=selected_classes,
        grid_layout=grid_layout
    )

    # 利用可能なクラスを取得
    unique_classes = sorted(np.unique(Y).astype(int).tolist())

    # UIを含む完全なHTMLページを生成
    html = generate_visualization_page(
        plot_html, dataset_name, user_id, sensor_name,
        unique_classes, selected_classes, grid_layout
    )

    return html


@app.route('/compare')
def compare():
    """複数センサーの比較ページ"""
    # クエリパラメータから比較対象を取得
    # 形式: sources=dataset1/user1/sensor1,dataset2/user2/sensor2
    sources_param = request.args.get('sources', '')
    grid_layout = request.args.get('grid', '2x2')
    selected_classes_param = request.args.get('classes', None)

    if not sources_param:
        return "No sources specified. Use ?sources=dataset/user/sensor,dataset/user/sensor", 400

    # ソースをパース
    sources = []
    for source in sources_param.split(','):
        parts = source.strip().split('/')
        if len(parts) == 3:
            sources.append({
                'dataset': parts[0],
                'user': parts[1],
                'sensor': parts[2]
            })

    if not sources:
        return "Invalid sources format", 400

    # クラスフィルタのパース
    selected_classes = None
    if selected_classes_param:
        try:
            selected_classes = [int(c) for c in selected_classes_param.split(',')]
        except ValueError:
            pass

    # グリッドレイアウトのパース
    rows, cols = map(int, grid_layout.split('x'))
    total_subplots = rows * cols

    # 各ソースからデータをロード
    all_data = []
    for source in sources[:total_subplots]:  # グリッドサイズに制限
        X, Y = load_sensor_data(source['dataset'], source['user'], source['sensor'])
        if X is not None and Y is not None:
            all_data.append({
                'X': X,
                'Y': Y,
                'dataset': source['dataset'],
                'user': source['user'],
                'sensor': source['sensor']
            })

    if not all_data:
        return "No valid data found for the specified sources", 404

    # 可視化を生成
    plot_html = create_comparison_visualization(
        all_data, selected_classes, grid_layout
    )

    # UIを含む完全なHTMLページを生成
    html = generate_comparison_page(
        plot_html, all_data, sources, selected_classes, grid_layout
    )

    return html


def main():
    parser = argparse.ArgumentParser(description='HAR Data Visualization Web App')
    parser.add_argument('--port', type=int, default=5000, help='Port number (default: 5000)')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug mode (default: enabled)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host address (default: 127.0.0.1)')

    args = parser.parse_args()

    # デバッグモードはデフォルトで有効（ホットリロード対応）
    debug_mode = not args.no_debug

    print("=" * 80)
    print("🌐 HAR Data Visualization Web App")
    print("=" * 80)
    print(f"  URL: http://{args.host}:{args.port}")
    print(f"  Data directory: {DATA_DIR.absolute()}")
    print(f"  Debug mode: {'ON' if debug_mode else 'OFF'} (Hot reload: {'enabled' if debug_mode else 'disabled'})")
    print(f"  Press Ctrl+C to stop the server")
    print("=" * 80)

    # サーバー起動（デフォルトでデバッグモード有効）
    app.run(host=args.host, port=args.port, debug=debug_mode, use_reloader=debug_mode)


if __name__ == '__main__':
    main()
