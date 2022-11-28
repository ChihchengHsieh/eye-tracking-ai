# eye-tracking-ai

# Experiments

Let's first get a trainable model.

## Exp1 (0.0114)

```
========================================For Training [All_label_testing]========================================
ModelSetup(name='All_label_testing', use_heatmaps=False, with_fixations=False, with_pupil=False, with_1st_third_fixations=False, with_2nd_third_fixations=False, with_rad_silence=False, with_rad_speaking=False, save_early_stop_model=True, record_training_performance=True, backbone='resnet18', optimiser='sgd', lr=0.001, weight_decay=1e-05, image_backbone_pretrained=True, heatmap_backbone_pretrained=True, image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=True, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=0, fusion_strategy='multiply', fusion_residule=False, gt_in_train_till=0, measure_test=True)
===============================================================================================================

# Best AP validation model has been saved to: [val_ar_0_0493_ap_0_0135_test_ar_0_0500_ap_0_0114_epoch36_11-07-2022 08-00-45_All_label_testing]
# Best AR validation model has been saved to: [val_ar_0_0619_ap_0_0099_test_ar_0_0569_ap_0_0114_epoch70_11-07-2022 14-08-34_All_label_testing]
# The final model has been saved to: [val_ar_0_0487_ap_0_0112_test_ar_0_0494_ap_0_0124_epoch100_11-07-2022 19-23-44_All_label_testing]

================================================================================================================

```

## Exp2 (0.0331)

```
========================================For Training [label_testing]========================================
ModelSetup(name='label_testing', use_heatmaps=False, with_fixations=False, with_pupil=False, with_1st_third_fixations=False, with_2nd_third_fixations=False, with_rad_silence=False, with_rad_speaking=False, save_early_stop_model=True, record_training_performance=True, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=1e-05, image_backbone_pretrained=True, heatmap_backbone_pretrained=True, image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=True, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=0, fusion_strategy='add', fusion_residule=False, gt_in_train_till=5, measure_test=True, eval_freq=10)
============================================================================================================

Best AP validation model has been saved to: [val_ar_0_1581_ap_0_0376_test_ar_0_1520_ap_0_0331_epoch20_11-13-2022 22-32-09_label_testing]
Best AR validation model has been saved to: [val_ar_0_1581_ap_0_0376_test_ar_0_1520_ap_0_0331_epoch20_11-13-2022 22-32-08_label_testing]
The final model has been saved to: [val_ar_0_1581_ap_0_0376_test_ar_0_1520_ap_0_0335_epoch20_11-13-2022 22-33-34_label_testing]

============================================================================================================
Using pretrained backbone. mobilenet_v3
label_testing will use mask, [64] layers.
[model]: 3,491,471
[model.backbone]: 1,258,848
[model.rpn]: 41,803
[model.roi_heads]: 2,190,820
[model.roi_heads.box_head]: 204,928
[model.roi_heads.box_head.fc6]: 200,768
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
[model.roi_heads.mask_head]: 1,917,952
Max AP on test: [0.0362]
```

## Exp3 (0.0415)

```
========================================For Training [label_testing]========================================
ModelSetup(name='label_testing', use_heatmaps=False, with_fixations=False, with_pupil=False, with_1st_third_fixations=False, with_2nd_third_fixations=False, with_rad_silence=False, with_rad_speaking=False, save_early_stop_model=True, record_training_performance=True, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=1e-05, image_backbone_pretrained=True, heatmap_backbone_pretrained=True, image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=True, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=0, fusion_strategy='add', fusion_residule=False, gt_in_train_till=5, measure_test=True, eval_freq=10)
============================================================================================================

Best AP validation model has been saved to: [val_ar_0_1657_ap_0_0379_test_ar_0_1620_ap_0_0415_epoch40_11-14-2022 11-48-15_label_testing]
Best AR validation model has been saved to: [val_ar_0_1657_ap_0_0379_test_ar_0_1620_ap_0_0415_epoch40_11-14-2022 11-48-14_label_testing]
The final model has been saved to: [val_ar_0_1657_ap_0_0356_test_ar_0_1600_ap_0_0401_epoch50_11-14-2022 13-28-38_label_testing]

============================================================================================================
Using pretrained backbone. mobilenet_v3
label_testing will use mask, [64] layers.
[model]: 3,491,471
[model.backbone]: 1,258,848
[model.rpn]: 41,803
[model.roi_heads]: 2,190,820
[model.roi_heads.box_head]: 204,928
[model.roi_heads.box_head.fc6]: 200,768
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
[model.roi_heads.mask_head]: 1,917,952
Max AP on test: [0.0416]
```

## Exp4

```
========================================For Training [label_testing]========================================
ModelSetup(name='label_testing', use_heatmaps=False, with_fixations=False, with_pupil=False, with_1st_third_fixations=False, with_2nd_third_fixations=False, with_rad_silence=False, with_rad_speaking=False, save_early_stop_model=True, record_training_performance=True, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=1e-05, image_backbone_pretrained=True, heatmap_backbone_pretrained=True, image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=True, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=0, fusion_strategy='add', fusion_residule=False, gt_in_train_till=5, measure_test=True, eval_freq=10)
============================================================================================================

Best AP validation model has been saved to: [val_ar_0_4594_ap_0_1900_test_ar_0_4590_ap_0_2022_epoch50_11-14-2022 16-34-52_label_testing]
Best AR validation model has been saved to: [val_ar_0_4594_ap_0_1900_test_ar_0_4590_ap_0_2022_epoch50_11-14-2022 16-34-52_label_testing]
The final model has been saved to: [val_ar_0_4594_ap_0_1900_test_ar_0_4590_ap_0_2022_epoch50_11-14-2022 16-35-26_label_testing]

============================================================================================================
Using pretrained backbone. mobilenet_v3
label_testing will use mask, [64] layers.
[model]: 3,491,471
[model.backbone]: 1,258,848
[model.rpn]: 41,803
[model.roi_heads]: 2,190,820
[model.roi_heads.box_head]: 204,928
[model.roi_heads.box_head.fc6]: 200,768
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
[model.roi_heads.mask_head]: 1,917,952
Max AP on test: [0.2022]
```

## Exp5

```
========================================For Training [label_testing]========================================
ModelSetup(name='label_testing', use_heatmaps=False, with_fixations=False, with_pupil=False, with_1st_third_fixations=False, with_2nd_third_fixations=False, with_rad_silence=False, with_rad_speaking=False, save_early_stop_model=True, record_training_performance=True, backbone='resnet18', optimiser='sgd', lr=0.001, weight_decay=1e-05, image_backbone_pretrained=True, heatmap_backbone_pretrained=True, image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=True, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=0, fusion_strategy='add', fusion_residule=False, gt_in_train_till=5, measure_test=True, eval_freq=10)
============================================================================================================

Best AP validation model has been saved to: [val_ar_0_3496_ap_0_1725_test_ar_0_2886_ap_0_1267_epoch50_11-14-2022 18-54-35_label_testing]
Best AR validation model has been saved to: [val_ar_0_3496_ap_0_1725_test_ar_0_2886_ap_0_1267_epoch50_11-14-2022 18-54-35_label_testing]
The final model has been saved to: [val_ar_0_3496_ap_0_1725_test_ar_0_2886_ap_0_1267_epoch50_11-14-2022 18-55-04_label_testing]

============================================================================================================
Using pretrained backbone. resnet18
label_testing will use mask, [64] layers.
[model]: 13,704,111
[model.backbone]: 11,471,488
[model.rpn]: 41,803
[model.roi_heads]: 2,190,820
[model.roi_heads.box_head]: 204,928
[model.roi_heads.box_head.fc6]: 200,768
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
[model.roi_heads.mask_head]: 1,917,952
Max AP on test: [0.1528]
```

## Exp6

```
========================================For Training [label_testing]========================================
ModelSetup(name='label_testing', use_heatmaps=False, with_fixations=False, with_pupil=False, with_1st_third_fixations=False, with_2nd_third_fixations=False, with_rad_silence=False, with_rad_speaking=False, save_early_stop_model=True, record_training_performance=True, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=1e-05, image_backbone_pretrained=True, heatmap_backbone_pretrained=True, image_size=256, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=True, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=0, fusion_strategy='add', fusion_residule=False, gt_in_train_till=5, measure_test=True, eval_freq=10)
============================================================================================================

Best AP validation model has been saved to: [val_ar_0_4016_ap_0_1718_test_ar_0_3987_ap_0_1804_epoch50_11-14-2022 21-36-49_label_testing]
Best AR validation model has been saved to: [val_ar_0_4016_ap_0_1718_test_ar_0_3987_ap_0_1804_epoch50_11-14-2022 21-36-49_label_testing]
The final model has been saved to: [val_ar_0_4016_ap_0_1718_test_ar_0_3987_ap_0_1804_epoch50_11-14-2022 21-37-25_label_testing]

============================================================================================================
Using pretrained backbone. mobilenet_v3
label_testing will use mask, [64] layers.
[model]: 3,491,471
[model.backbone]: 1,258,848
[model.rpn]: 41,803
[model.roi_heads]: 2,190,820
[model.roi_heads.box_head]: 204,928
[model.roi_heads.box_head.fc6]: 200,768
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
[model.roi_heads.mask_head]: 1,917,952
Max AP on test: [0.1804]
```

## Exp7 Top 5 labels (0.26) - baseline

```
========================================For Training [label_testing]========================================
ModelSetup(name='label_testing', use_heatmaps=False, with_fixations=False, with_pupil=False, with_1st_third_fixations=False, with_2nd_third_fixations=False, with_rad_silence=False, with_rad_speaking=False, save_early_stop_model=True, record_training_performance=True, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=1e-05, image_backbone_pretrained=True, heatmap_backbone_pretrained=True, image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=True, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=0, fusion_strategy='add', fusion_residule=False, gt_in_train_till=10, measure_test=True, eval_freq=10)
============================================================================================================

Best AP validation model has been saved to: [val_ar_0_5178_ap_0_2403_test_ar_0_5277_ap_0_2606_epoch50_11-17-2022 08-48-30_label_testing]
Best AR validation model has been saved to: [val_ar_0_5695_ap_0_2207_test_ar_0_5617_ap_0_2566_epoch40_11-17-2022 06-47-41_label_testing]
The final model has been saved to: [val_ar_0_4232_ap_0_2044_test_ar_0_4178_ap_0_2282_epoch181_11-18-2022 17-34-04_label_testing]

============================================================================================================
Using pretrained backbone. mobilenet_v3
label_testing will use mask, [64] layers.
[model]: 3,491,471
[model.backbone]: 1,258,848
[model.rpn]: 41,803
[model.roi_heads]: 2,190,820
[model.roi_heads.box_head]: 204,928
[model.roi_heads.box_head.fc6]: 200,768
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
[model.roi_heads.mask_head]: 1,917,952
Max AP on test: [0.2606]
Labels used:
 ['Pulmonary edema', 'Enlarged cardiac silhouette', 'Consolidation', 'Atelectasis', 'Pleural abnormality']
```

### Exp8 (Atelectasis only) - [0.0096]

```
========================================For Training [label_testing]========================================
ModelSetup(name='label_testing', use_heatmaps=False, with_fixations=False, with_pupil=False, with_1st_third_fixations=False, with_2nd_third_fixations=False, with_rad_silence=False, with_rad_speaking=False, save_early_stop_model=True, record_training_performance=True, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=1e-05, image_backbone_pretrained=True, heatmap_backbone_pretrained=True, image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=True, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=0, fusion_strategy='add', fusion_residule=False, gt_in_train_till=10, measure_test=True, eval_freq=10)
============================================================================================================

Best AP validation model has been saved to: [val_ar_0_0783_ap_0_0102_test_ar_0_0769_ap_0_0096_epoch10_11-19-2022 03-22-24_label_testing]
Best AR validation model has been saved to: [val_ar_0_0783_ap_0_0102_test_ar_0_0769_ap_0_0096_epoch10_11-19-2022 03-22-23_label_testing]
The final model has been saved to: [val_ar_0_0420_ap_0_0076_test_ar_0_0346_ap_0_0035_epoch100_11-19-2022 16-24-15_label_testing]

============================================================================================================
Using pretrained backbone. mobilenet_v3
label_testing will use mask, [64] layers.
[model]: 3,489,911
[model.backbone]: 1,258,848
[model.rpn]: 41,803
[model.roi_heads]: 2,189,260
[model.roi_heads.box_head]: 204,928
[model.roi_heads.box_head.fc6]: 200,768
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 650
[model.roi_heads.mask_head]: 1,917,952
Max AP on test: [0.0096]
Labels used:
 ['Atelectasis']
```

### Exp9 (top6)

```

========================================For Training [label_testing]========================================
ModelSetup(name='label_testing', use_heatmaps=False, with_fixations=False, with_pupil=False, with_1st_third_fixations=False, with_2nd_third_fixations=False, with_rad_silence=False, with_rad_speaking=False, save_early_stop_model=True, record_training_performance=True, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=1e-05, image_backbone_pretrained=True, heatmap_backbone_pretrained=True, image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=True, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=0, fusion_strategy='add', fusion_residule=False, gt_in_train_till=10, measure_test=True, eval_freq=10)
============================================================================================================

Best AP validation model has been saved to: [val_ar_0_5650_ap_0_2376_test_ar_0_5607_ap_0_2611_epoch80_11-20-2022 09-39-54_label_testing]
Best AR validation model has been saved to: [val_ar_0_5923_ap_0_2304_test_ar_0_5914_ap_0_2604_epoch40_11-20-2022 01-40-56_label_testing]
The final model has been saved to: [val_ar_0_4789_ap_0_2030_test_ar_0_4949_ap_0_2373_epoch100_11-20-2022 13-38-27_label_testing]

============================================================================================================
Using pretrained backbone. mobilenet_v3
label_testing will use mask, [64] layers.
[model]: 3,491,861
[model.backbone]: 1,258,848
[model.rpn]: 41,803
[model.roi_heads]: 2,191,210
[model.roi_heads.box_head]: 204,928
[model.roi_heads.box_head.fc6]: 200,768
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 2,275
[model.roi_heads.mask_head]: 1,917,952
Max AP on test: [0.2612]
Labels used: ['Pulmonary edema', 'Enlarged cardiac silhouette', 'Consolidation', 'Atelectasis', 'Pleural abnormality', 'Support devices']
```
the `supported device` label here is found not detectable. Therefore, we will try to train a model only for this label.

# Exp10 (only supported device)

```
========================================For Training [label_testing]========================================
ModelSetup(name='label_testing', use_heatmaps=False, with_fixations=False, with_pupil=False, with_1st_third_fixations=False, with_2nd_third_fixations=False, with_rad_silence=False, with_rad_speaking=False, save_early_stop_model=True, record_training_performance=True, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=1e-05, image_backbone_pretrained=True, heatmap_backbone_pretrained=True, image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=True, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=0, fusion_strategy='add', fusion_residule=False, gt_in_train_till=10, measure_test=True, eval_freq=10)
============================================================================================================

Best AP validation model has been saved to: [val_ar_0_0000_ap_0_0000_test_ar_0_0000_ap_0_0000_epoch1_11-20-2022 22-00-19_label_testing]
Best AR validation model has been saved to: [val_ar_0_0000_ap_0_0000_test_ar_0_0000_ap_0_0000_epoch1_11-20-2022 22-00-18_label_testing]
The final model has been saved to: [val_ar_0_0000_ap_0_0000_test_ar_0_0000_ap_0_0000_epoch100_11-21-2022 10-38-44_label_testing]

============================================================================================================
Using pretrained backbone. mobilenet_v3
label_testing will use mask, [64] layers.
[model]: 3,489,911
[model.backbone]: 1,258,848
[model.rpn]: 41,803
[model.roi_heads]: 2,189,260
[model.roi_heads.box_head]: 204,928
[model.roi_heads.box_head.fc6]: 200,768
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 650
[model.roi_heads.mask_head]: 1,917,952
Max AP on test: [0.0000]
Labels used:
 ['Support devices']
```


# Exp 11(100epoch not enough)

```
========================================For Training [label_testing]========================================
ModelSetup(name='label_testing', use_heatmaps=False, with_fixations=False, with_pupil=False, with_1st_third_fixations=False, with_2nd_third_fixations=False, with_rad_silence=False, with_rad_speaking=False, save_early_stop_model=True, record_training_performance=True, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=1e-05, image_backbone_pretrained=True, heatmap_backbone_pretrained=True, image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=True, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=0, fusion_strategy='add', fusion_residule=False, gt_in_train_till=10, measure_test=True, eval_freq=10)
============================================================================================================

Best AP validation model has been saved to: [val_ar_0_2357_ap_0_0475_test_ar_0_2608_ap_0_0488_epoch40_11-21-2022 20-46-57_label_testing]
Best AR validation model has been saved to: [val_ar_0_2562_ap_0_0439_test_ar_0_2728_ap_0_0419_epoch90_11-22-2022 06-57-45_label_testing]
The final model has been saved to: [val_ar_0_2463_ap_0_0393_test_ar_0_2829_ap_0_0399_epoch100_11-22-2022 09-01-15_label_testing]

============================================================================================================
Using pretrained backbone. mobilenet_v3
label_testing will use mask, [64] layers.
[model]: 3,491,861
[model.backbone]: 1,258,848
[model.rpn]: 41,803
[model.roi_heads]: 2,191,210
[model.roi_heads.box_head]: 204,928
[model.roi_heads.box_head.fc6]: 200,768
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 2,275
[model.roi_heads.mask_head]: 1,917,952
Max AP on test: [0.0488]
Labels used:
 ['Groundglass opacity', 'Pulmonary edema', 'Enlarged cardiac silhouette', 'Consolidation', 'Atelectasis', 'Pleural abnormality']
```


# Exp12 (5andLMandGO)

```
========================================For Training [label_testing]========================================
ModelSetup(name='label_testing', use_heatmaps=False, with_fixations=False, with_pupil=False, with_1st_third_fixations=False, with_2nd_third_fixations=False, with_rad_silence=False, with_rad_speaking=False, save_early_stop_model=True, record_training_performance=True, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=1e-05, image_backbone_pretrained=True, heatmap_backbone_pretrained=True, image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=True, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=0, fusion_strategy='add', fusion_residule=False, gt_in_train_till=10, measure_test=True, eval_freq=10)
============================================================================================================

Best AP validation model has been saved to: [val_ar_0_1868_ap_0_0314_test_ar_0_2216_ap_0_0421_epoch10_11-26-2022 20-33-47_label_testing]
Best AR validation model has been saved to: [val_ar_0_1868_ap_0_0314_test_ar_0_2216_ap_0_0421_epoch10_11-26-2022 20-33-46_label_testing]
The final model has been saved to: [val_ar_0_0907_ap_0_0103_test_ar_0_1007_ap_0_0123_epoch200_11-28-2022 08-46-07_label_testing]

============================================================================================================
Using pretrained backbone. mobilenet_v3
label_testing will use mask, [64] layers.
[model]: 3,492,251
[model.backbone]: 1,258,848
[model.rpn]: 41,803
[model.roi_heads]: 2,191,600
[model.roi_heads.box_head]: 204,928
[model.roi_heads.box_head.fc6]: 200,768
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 2,600
[model.roi_heads.mask_head]: 1,917,952
Max AP on test: [0.0421]
Labels used:
 ['Lung nodule or mass', 'Groundglass opacity', 'Pulmonary edema', 'Enlarged cardiac silhouette', 'Consolidation', 'Atelectasis', 'Pleural abnormality']
```


![image](https://user-images.githubusercontent.com/37566901/204273405-7954356a-55d5-4378-8e47-8c9659050017.png)


<!-- ![image](https://user-images.githubusercontent.com/37566901/203262882-7732c348-b93f-4a7f-be7f-898681dbffef.png)

the 6th label alwas have -1 here, why? Is it a bug? test it with 7 diseasese. -->
