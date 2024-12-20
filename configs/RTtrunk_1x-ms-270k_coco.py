default_scope = 'mmdet'
dataset_type = 'CocoDataset' 
backend = 'pillow'
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'projects.RT-Trunk.sparseinst',
        'mmpretrain.models',
    ])

launcher = 'none'

# Modify dataset related settings
METAINFO = {
    'classes': ('tree',),
    'palette': [(185, 90, 200),]
}

num_classes = 1
num_masks = 30
img_scale = (1024, 1024)
max_batch_size = 1
lr_base = 0.0001
logger_intervl = 100


fileFolder = 'ForestTreeTrunkInsSeg_MixDatasets/'
project_name = "mmDetTreeTrunkConcat"
model_name = 'RTtrunk_1x-ms-270k'

data_root = '../../datasets_v1.0/' + fileFolder 
result_save_dir = 'work_dirs/' + fileFolder + 'TreeTrunkConcat/' + model_name

data_root_CanaTree100 = data_root  + 'CanaTree100'
data_root_DatasetNinjaTreeBinaryInsSeg = data_root  + 'DatasetNinjaTreeBinaryInsSeg'
data_root_FinnWoodlandsInsSeg = data_root  + 'FinnWoodlandsInsSeg'
data_root_ForestTreeTrunkMC100 = data_root  + 'ForestTreeTrunkMC100'
data_root_ForestTreeTrunkNEFU1095 = data_root  + 'ForestTreeTrunkNEFU1095'
data_root_SynthTree43kWYZPart308 = data_root  + 'SynthTree43kWYZPart308'
data_root_TreeTrunkWYZNEFU331 = data_root  + 'TreeTrunkWYZNEFU331'

data_val_root = data_root + 'ForestTreeTrunkMixVal/'
data_test_root = data_root + 'ForestTreeTrunkMixTest/'
test_dataset = 'test2017'
val_dataset = 'val2017'

#load_from = None
load_from = './projects/RT-Trunk/pretrained_weights/sparse_inst_r50_giam_aug_2b7d68.pth'
checkpoint = './projects/RT-Trunk/pretrained_weights/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'

model = dict(
    backbone=dict(
        type='mmpretrain.ConvNeXt',
        arch='tiny',
        out_indices=[1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint,
            prefix='backbone.')),
    criterion=dict(
        assigner=dict(alpha=0.8, beta=0.2, type='SparseInstMatcher'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=2.0,
            reduction='sum',
            type='FocalLoss',
            use_sigmoid=True),
        loss_dice=dict(
            eps=5e-05,
            loss_weight=2.0,
            reduction='sum',
            type='DiceLoss',
            use_sigmoid=True),
        loss_mask=dict(
            loss_weight=5.0,
            reduction='mean',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        loss_obj=dict(
            loss_weight=1.0,
            reduction='mean',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        num_classes=num_classes,
        type='SparseInstCriterion'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        in_channels=258,
        num_groups=4,
        ins_conv=4,
        ins_dim=256,
        kernel_dim=128,
        mask_conv=4,
        mask_dim=256,
        num_classes=num_classes,
        num_masks=num_masks,
        output_iam=False,
        scale_factor=2.0,
        type='GroupIAMDecoder'),
    encoder=dict(
        in_channels=[
            192,
            384,
            768,
        ],
        out_channels=256,
        type='InstanceContextEncoder_CBAM'),
    test_cfg=dict(mask_thr_binary=0.45, score_thr=0.005),
    type='SparseInst')

backend_args = None

train_pipeline = [
    dict(
        backend_args=None, imdecode_backend='pillow',
        type='LoadImageFromFile'),
    dict(
        poly2mask=False,
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True),
    dict(
        backend='pillow',
        keep_ratio=True,
        scales=[
            (800, 1024,),
            (832, 1024,),
            (864, 1024,),
            (896, 1024,),
            (928, 1024,),
            (960, 1024,),
            (992, 1024,),
            (1024, 1024,),
        ],
        type='RandomChoiceResize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(
        backend_args=None, imdecode_backend='pillow',
        type='LoadImageFromFile'),
    dict(backend='pillow', keep_ratio=True, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]

data_CanaTree100 = dict(
    type=dataset_type,
    metainfo=METAINFO,
    data_root=data_root_CanaTree100,
    ann_file='annotations/instances_train2017.json',
    data_prefix=dict(img='train2017/JPEGImages/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)
data_DatasetNinjaTreeBinaryInsSeg = dict(
    type=dataset_type,
    metainfo=METAINFO,
    data_root=data_root_DatasetNinjaTreeBinaryInsSeg,
    ann_file='annotations/instances_train2017.json',
    data_prefix=dict(img='train2017/JPEGImages/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)
data_FinnWoodlandsInsSeg = dict(
    type=dataset_type,
    metainfo=METAINFO,
    data_root=data_root_FinnWoodlandsInsSeg,
    ann_file='annotations/instances_train2017.json',
    data_prefix=dict(img='train2017/JPEGImages/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)
data_ForestTreeTrunkMC100 = dict(
    type=dataset_type,
    metainfo=METAINFO,
    data_root=data_root_ForestTreeTrunkMC100,
    ann_file='annotations/instances_train2017.json',
    data_prefix=dict(img='train2017/JPEGImages/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)
data_ForestTreeTrunkNEFU1095 = dict(
    type=dataset_type,
    metainfo=METAINFO,
    data_root=data_root_ForestTreeTrunkNEFU1095,
    ann_file='annotations/instances_train2017.json',
    data_prefix=dict(img='train2017/JPEGImages/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)
data_SynthTree43kWYZPart308 = dict(
    type=dataset_type,
    metainfo=METAINFO,
    data_root=data_root_SynthTree43kWYZPart308,
    ann_file='annotations/instances_train2017.json',
    data_prefix=dict(img='train2017/JPEGImages/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)
data_TreeTrunkWYZNEFU331 = dict(
    type=dataset_type,
    metainfo=METAINFO,
    data_root=data_root_TreeTrunkWYZNEFU331,
    ann_file='annotations/instances_train2017.json',
    data_prefix=dict(img='train2017/JPEGImages/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline,
    backend_args=backend_args)

train_dataloader = dict(
    batch_size=max_batch_size,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=True, type='InfiniteSampler'),
    batch_sampler=dict(_scope_='mmdet', type='AspectRatioBatchSampler'),
    dataset=dict(
        _scope_='mmdet',
        type='ConcatDataset',
        datasets=[data_CanaTree100, data_DatasetNinjaTreeBinaryInsSeg, data_FinnWoodlandsInsSeg, data_ForestTreeTrunkMC100, data_SynthTree43kWYZPart308,data_ForestTreeTrunkNEFU1095, data_TreeTrunkWYZNEFU331],
    ))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'),
    dataset=dict(
        _scope_='mmdet',
        type=dataset_type,
        metainfo=METAINFO,
        data_root=data_val_root,
        ann_file='annotations/instances_' + val_dataset + '.json',
        data_prefix=dict(img=val_dataset+ '/JPEGImages/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args)
    )

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'),
    dataset=dict(
        _scope_='mmdet',
        type=dataset_type,
        metainfo=METAINFO,
        data_root=data_test_root,
        ann_file='annotations/instances_' + test_dataset + '.json',
        data_prefix=dict(img=test_dataset + '/JPEGImages/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args)
    )

val_evaluator = dict(
    _scope_='mmdet',
    ann_file=data_val_root + 'annotations/instances_' + val_dataset + '.json',
    backend_args=None,
    format_only=False,
    metric='segm',
    type='CocoMetric')

test_evaluator = dict(
    _scope_='mmdet',
    ann_file=data_test_root + 'annotations/instances_' + test_dataset + '.json',
    backend_args=None,
    format_only=False,
    metric='segm',
    type='CocoMetric')

train_cfg = dict(max_iters=270000, type='IterBasedTrainLoop', val_interval=10000)
val_cfg = dict(_scope_='mmdet', type='ValLoop')
test_cfg = dict(_scope_='mmdet', type='TestLoop')

optim_wrapper = dict(
    _scope_='mmdet',
    optimizer=dict(lr=lr_base, type='AdamW', weight_decay=0.05),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=270000,
        gamma=0.1,
        milestones=[
            200000,
            250000,
        ],
        type='MultiStepLR'),
]

auto_scale_lr = dict(base_batch_size=8, enable=True)

default_hooks = dict(
    checkpoint=dict(
        _scope_='mmdet',
        by_epoch=False,
        interval=10000,
        max_keep_ckpts=5,
        type='CheckpointHook'),
    logger=dict(_scope_='mmdet', interval=100, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmdet', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmdet', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmdet', type='IterTimerHook'),
    visualization=dict(_scope_='mmdet', type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend'), ]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    save_dir=result_save_dir,
    name='visualizer')

log_processor = dict(
    _scope_='mmdet', by_epoch=False, type='LogProcessor', window_size=50)

log_level = 'INFO'
resume = False

