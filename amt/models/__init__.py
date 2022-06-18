#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from amt.models.OTGT import BertClf
from amt.models.OIOT import MultimodalBertClf
from amt.models.AMT_model import DoubleMultimodalBertClf

MODELS = {
    "OTGT": BertClf, #OT+GT
    "OIOT": MultimodalBertClf, #OI+OT
    "AMT":DoubleMultimodalBertClf   
}


def get_model(args):
    return MODELS[args.model](args)
