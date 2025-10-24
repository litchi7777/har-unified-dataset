"""
データセット前処理モジュール

使用可能なプリプロセッサを登録・管理
"""

from typing import Dict, Type
from .base import BasePreprocessor

# プリプロセッサの登録辞書
PREPROCESSORS: Dict[str, Type[BasePreprocessor]] = {}


def register_preprocessor(name: str):
    """
    プリプロセッサを登録するデコレータ

    使用例:
        @register_preprocessor('dsads')
        class DSADSPreprocessor(BasePreprocessor):
            ...
    """
    def decorator(cls: Type[BasePreprocessor]):
        PREPROCESSORS[name] = cls
        return cls
    return decorator


def get_preprocessor(name: str) -> Type[BasePreprocessor]:
    """
    登録されたプリプロセッサを取得

    Args:
        name: データセット名

    Returns:
        プリプロセッサクラス

    Raises:
        KeyError: データセットが登録されていない場合
    """
    if name not in PREPROCESSORS:
        available = ', '.join(PREPROCESSORS.keys())
        raise KeyError(
            f"Preprocessor '{name}' not found. "
            f"Available preprocessors: {available}"
        )
    return PREPROCESSORS[name]


def list_preprocessors() -> list:
    """
    利用可能なプリプロセッサの一覧を取得

    Returns:
        データセット名のリスト
    """
    return list(PREPROCESSORS.keys())


# プリプロセッサを自動インポート
# 新しいデータセットを追加する際は、ここにインポートを追加
try:
    from .dsads import DSADSPreprocessor
except ImportError:
    pass

try:
    from .mhealth import MHEALTHPreprocessor
except ImportError:
    pass

try:
    from .openpack import OpenPackPreprocessor
except ImportError:
    pass


__all__ = [
    'BasePreprocessor',
    'register_preprocessor',
    'get_preprocessor',
    'list_preprocessors',
    'PREPROCESSORS',
]
