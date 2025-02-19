# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ['build_neck']


def build_neck(config):
    from .db_fpn import DBFPN, RSEFPN, LKPAN
    from .east_fpn import EASTFPN
    from .sast_fpn import SASTFPN
    from .rnn import SequenceEncoder
    from .pg_fpn import PGFPN
    from .table_fpn import TableFPN
    from .fpn import FPN
    from .fce_fpn import FCEFPN
    from .pren_fpn import PRENFPN
    from .csp_pan import CSPPAN
    from .ct_fpn import CTFPN
    from .fpn_unet import FPN_UNet
    from .rf_adaptor import RFAdaptor
    from .wff_fpn import WFFFPN
    from .wff_fpn_2 import WFFFPN
    support_dict = [
        'FPN', 'FCEFPN', 'LKPAN', 'DBFPN', 'RSEFPN', 'EASTFPN', 'SASTFPN',
        'SequenceEncoder', 'PGFPN', 'TableFPN', 'PRENFPN', 'CSPPAN', 'CTFPN',
        'RFAdaptor', 'FPN_UNet', 'WFFFPN'
    ]

    module_name = config.pop('name')
    assert module_name in support_dict, Exception('neck only support {}'.format(
        support_dict))

    module_class = eval(module_name)(**config)
    return module_class
