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

![image](https://user-images.githubusercontent.com/37566901/202744583-9f5e8fb4-5196-4a75-a1d2-1435672fe3b7.png)
![image](https://user-images.githubusercontent.com/37566901/202744600-0689da51-72f3-49e5-a6bf-9d1803d23fdd.png)



### Exp8
