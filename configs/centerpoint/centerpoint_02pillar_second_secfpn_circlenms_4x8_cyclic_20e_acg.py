_base_ = ['./centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_acg.py']

model = dict(test_cfg=dict(pts=dict(nms_type='circle')))
