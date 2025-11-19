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

try:
    from .nhanes_pax import NHANESPreprocessor
except ImportError:
    pass

try:
    from .forthtrace import ForthtracePreprocessor
except ImportError:
    pass

try:
    from .har70plus import Har70plusPreprocessor
except ImportError:
    pass

try:
    from .harth import HarthPreprocessor
except ImportError:
    pass

try:
    from .realworld import RealWorldPreprocessor
except ImportError:
    pass
  
try:
    from .lara import LaraPreprocessor
except ImportError:
    pass
  
try:
    from .realdisp import RealDispPreprocessor
except ImportError:
    pass

try:
    from .harth import HarthPreprocessor
except ImportError:
    pass

try:
    from .mex import MexPreprocessor
except ImportError:
    pass
  
    
try:
    from .pamap2 import PAMAP2Preprocessor

except ImportError:
    pass
  
try:
    from .opportunity import OpportunityPreprocessor
except ImportError:
    pass

try:
    from .uschad import USCHADPreprocessor
except ImportError:
    pass

try:
    from .selfback import SelfBackPreprocessor
except ImportError:
    pass

try:
    from .paal import PAALPreprocessor
except ImportError:
    pass
  
try:
    from .hhar import HHARPreprocessor
except ImportError:
    pass
  
try:
    from .wisdm import WISDMPreprocessor
except ImportError:
    pass

try:
    from .tmd import TMDPreprocessor
except ImportError:
    pass

try:
    from .ward import WARDPreprocessor
except ImportError:
    pass

try:
    from .hmp import HMPPreprocessor
except ImportError:
    pass

try:
    from .capture24 import Capture24Preprocessor
except ImportError:
    pass

try:
    from .chad import CHADPreprocessor
except ImportError:
    pass

try:
    from .imsb import IMSBPreprocessor
except ImportError:
    pass

try:
    from .motionsense import MotionSensePreprocessor
except ImportError:
    pass

try:
    from .imwsha import IMWSHAPreprocessor
except ImportError:
    pass

try:
    from .sbrhapt import SBRHAPTPreprocessor
except ImportError:
    pass


__all__ = [
    'BasePreprocessor',
    'register_preprocessor',
    'get_preprocessor',
    'list_preprocessors',
    'PREPROCESSORS',
]
