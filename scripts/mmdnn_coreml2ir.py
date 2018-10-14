import os
import sys as _sys

from mmdnn.conversion.coreml.coreml_parser import CoremlParser
from coremltools.models import MLModel as _MLModel

def main():
                
    loaded_model_ml = _MLModel('result/posenet257.mlmodel')
    coremlParser = CoremlParser(loaded_model_ml)
    coremlParser.run('result/posenet257_ir')



if __name__ == '__main__':
    main()
