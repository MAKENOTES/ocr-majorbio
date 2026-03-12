标注数据
    1、统一标注规则，让模型能够提取共性（特征）。比如：框都紧贴文字，不要出现有些紧贴有些有较大空白。
    2、易忽略点，关注小面积框，要符合规范，小面积框同样是一个样本。[]()
    3、前期可以去掉一些特殊样本，特别是实际垂直业务场景不会使用的。
    4、可以通过 TensorBoard 的批次在相关loss 表现，可以定位优化标注的数据，或者放弃某些特殊的样本
训练优化方案[]()
    模型构造优化、 超参的优化 、算法的优化 、训练方式的优化
    训练批次1
        学习率：1e-4    lambda_iou：1   shrink：0.4   epoch：200
        结果：总损失0.12xx  score 0.000x  geo 0.12xx，   整体框左右很多不准，每页可能10个左右，很多框空白。
        总结：geo的损失下不来。
    训练批次2
        学习率：3e-5    lambda_iou：3   shrink：0.2   epoch：500
        结果：总损失0.12xx  score  0.004x  geo 0.04xx，   整体框准很多， 每页可能5个左右，基本消除了框空白的现象。
        总结：感觉 score和geo还有下降空间，加大epoch。
    训练批次3
        学习率：1e-5    lambda_iou：3   shrink：0.2   epoch：1200（1180_0.1116.pth）+500(0.0968.pth) +1000（Loss: 0.0654 (Score: 0.0055, Geo: 0.0199)）
        结果：总损失0.09xx  score  0.006x  geo 0.03xx，   
        总结：最后追加的1000， 基本达到预期。感觉还可以追加  后续可以尝试


接下来可尝试的计划
    1、调整权重3--->2，   学习率：1e-5    lambda_iou：2   shrink：0.2   epoch：1700
    2、调整iou 算法  diou， 1e-5    lambda_iou：3   shrink：0.2   epoch：1700
    3、两个阶段训练，
       先1e-4 权重3--->1  得到小的score， 学习率：1e-5    lambda_iou：3   shrink：0.2   epoch：1700
       然后调追加训练 整权重3--->2  学习率1e-5  降低geo， 1e-5    lambda_iou：3   shrink：0.2   epoch：1700
    4、调整模型构建

20260310   更换了resnet18 和 图片输入分辨率 640 ， 目的是为了提高在cpu上的推理速度
    此次代码改动，主要调整三处，
        第一个 新的模型文件，
        第二个 新的推理文件，
        第三个 图片分辨率1024到640  train文件 和 推理文件，

    训练完成后，存在两个问题
        出现两行文字一个框 （预估此问题需要重新训练）
        单字或单个符号 漏调 （新增v3推理改善很多）


    已保存早停时的模型: detection/checkpoints/detection_epoch_248_early_stop.pth
    ============================================================
    训练因早停在 Epoch 248 结束
    最佳模型: Epoch 208, Val Loss: 0.01717
    Top-3 模型:
      1. best_detection_epoch_208_loss_0.01717.pth (Loss: 0.01717)
      2. best_detection_epoch_207_loss_0.01738.pth (Loss: 0.01738)
      3. best_detection_epoch_240_loss_0.01782.pth (Loss: 0.01782)


20260311   ground_truth.py shrunk_quad = shrink_polygon(scaled_quad, r=0.3)   0.2换成0.3 让行间有更明确的间隔  ， 解决出现两行文字一个框
    0.2换成0.3【shrink_polygon(scaled_quad, r=0.3) 】，并没有解决问题。 但是 训练没有早停止  而且 loss 降到了0.00831。
    ============================================================
    训练完成，共 500 个 Epoch
    最佳模型: Epoch 475, Val Loss: 0.00831
    Top-3 模型:
    1. best_detection_epoch_475_loss_0.00831.pth (Loss: 0.00831)
    2. best_detection_epoch_484_loss_0.00836.pth (Loss: 0.00836)
    3. best_detection_epoch_500_loss_0.00836.pth (Loss: 0.00836)


20260311  ground_truth 替换， （原来的文件备份给resnet50 ,命名为ground_truth_resnet50  并恢复缩放03成0.2），   目的是解决640分辨率下  两行文字一个框问题
    替换后模型效果很差 无法使用  【待处理】

    ============================================================
    训练完成，共 500 个 Epoch
    最佳模型: Epoch 492, Val Loss: 0.01948
    Top-3 模型:
      1. best_detection_epoch_492_loss_0.01948.pth (Loss: 0.01948)
      2. best_detection_epoch_489_loss_0.01950.pth (Loss: 0.01950)
      3. best_detection_epoch_494_loss_0.01951.pth (Loss: 0.01951)


