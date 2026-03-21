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
    结果：替换后模型效果很差 无法使用  TODO

    ============================================================
    训练完成，共 500 个 Epoch
    最佳模型: Epoch 492, Val Loss: 0.01948
    Top-3 模型:
      1. best_detection_epoch_492_loss_0.01948.pth (Loss: 0.01948)
      2. best_detection_epoch_489_loss_0.01950.pth (Loss: 0.01950)
      3. best_detection_epoch_494_loss_0.01951.pth (Loss: 0.01951)

20260312   计划使用 moble模型架构, 目的优化速度
    效果比resnet18 差一点，
    ============================================================
    ★ 早停触发！连续 40 次验证无改善
      最佳 Val Loss: 0.02701 (Epoch 247)
      当前 Val Loss: 0.04043 (Epoch 287)
      建议使用 Epoch 247 的模型进行推理
    ============================================================

    已保存早停时的模型: detection/checkpoints/detection_epoch_287_early_stop.pth

    ============================================================
    训练因早停在 Epoch 287 结束
    最佳模型: Epoch 247, Val Loss: 0.02701
    Top-3 模型:
      1. best_detection_epoch_247_loss_0.02701.pth (Loss: 0.02701)
      2. best_detection_epoch_248_loss_0.02754.pth (Loss: 0.02754)
      3. best_detection_epoch_235_loss_0.02756.pth (Loss: 0.02756)

20260313  调整 mobilenet，  目的解决 loss下降不够和应该是两行框 ，只有一行框的问题。
        使用分辨率 768  以及调整 BATCH_SIZE 4 --》 8,   基本解决（应该是两行框 ，只有一行框的问题），loss 下降还ok，
        但是还是存在一些问题， 表格线条被框出来了。

重要总结：
    1、造成应该是两行框 ，只有一行框的问题 ，以及 小的单个符号没有标记。  都是因为分辨率问题，从1024 降到 640  就会出现。 如果是768  此问题会被改善很多（造成应该是两行框 ，只有一行框的问题， 其实时代码问题，看20260320的总结）。
    2、mobilenet 静态量化，不推荐， 难以提高速度和降低准确， 后面可以再尝试量化。

20260315 mobilenet 使用1024 分辨率，  验证20260313 的问题。
    下一步
        1、预计处理训练脚步  ，保存时选择  train_val 还是 validated_val 来作为保存标准。另外，推理的分两批置信度的逻辑考虑曲调。
        2、resnet18， 可以考虑训练，resnet18 时可以进行静态量化，性能和准确率，还需要与mobilenet验证。
    后续训练推荐，分辨率：选择1024 ，不然效果很差。
        1024 训练结果 best_detection_epoch_368_train_loss_0.01697_mobilenet1024，  出框基本没啥问题。
        640  训练结果 best_detection_epoch_441_train_loss_0.01042_mobilenet_640，  出框问题多，漏框，本应该两个框，只有一个框。


20260320
    改动
        input_size 640、
        resnet18、
        原来  max_d = (min_side - 1.0) / 2.0   其中1.0 时常量，改成  max_d = min_side * 0.45
    1、根据缩放策略 一次 ,  结果出现少部分表格数据出框不好，不在乎表格的可以使用。
        best_detection_epoch_155_loss_0.02183_resnet_640
    2、缩放统一 0.2  一次, 结果基本ok。
        best_detection_epoch_155_loss_0.02183_resnet_640



    总结：
        1、input_size 由1024改成640  原来  max_d = (min_side - 1.0) / 2.0   其中1.0 时常量，改成  max_d = min_side * 0.45 可以适配640。解决上下两行仅出一个框问题


20260321   resnet50 ，1024 分辨率 更多指标训练脚步  一次

    (.venv) weiqiang@weiqiang:~/AI/ocr-majorbio$ python detection_quantize_int8_v2.py
    ============================================================
    EAST 模型 INT8 静态量化 v2
    ============================================================
      输入模型:     best_detection.onnx
      输出模型:     best_detection_int8_static.onnx
      量化格式:     QOperator (U8S8) → 文件体积缩小
      校准目录:     detection/data/val_images
      校准样本数:   100
      Per-channel:  False
      Reduce-range: False
      校准方法:     MinMax
      输入尺寸:     640x640

    [1/6] 原始 FP32 模型大小: 47.6 MB

    [2/6] 分析模型结构，定位敏感输出节点...
      自动排除输出头节点: ['node_conv2d_23', 'node_sigmoid', 'node_conv2d_24']
        → 这些节点保留 FP32 精度，避免 score/geo 量化误差

    [3/6] 模型预处理 (shape inference + optimization)...
      预处理模型: best_detection.onnx → best_detection_prep.onnx
      ✓ 预处理完成
      自动排除输出头节点: ['node_conv2d_23', 'node_sigmoid', 'node_conv2d_24']
        → 这些节点保留 FP32 精度，避免 score/geo 量化误差

    [4/6] 准备校准数据...
      校准图像: 100 张 (来自 detection/data/val_images)
      成功加载 100 张校准图像

    [5/6] 执行 INT8 静态量化 (QOperator)...
      ✓ 量化完成！ (耗时 131.7s)
      FP32 模型:  47.6 MB
      INT8 模型:  13.5 MB
      压缩比:     3.51x

    [6/6] 验证 FP32 vs INT8 输出一致性...
      score_map 最大误差:   2.7082e-02
      score_map 平均误差:   1.3599e-05
      score_map 相关系数:   0.898516
      geo_map   最大误差:   6.7916e-01
      geo_map   平均误差:   6.2656e-02
      ⚠ 精度损失较大，建议：
        1. 尝试 per-channel 量化 (--per_channel)
        2. 尝试 Entropy 校准 (--calibrate_method entropy)
        3. 增加校准样本 (--calib_num 200)
        4. 非 VNNI CPU 尝试 --reduce_range

    [Benchmark] 速度对比 (CPU, 5次预热 + 20次计时)...
      ONNX FP32:           1780.2 ms/张
      ONNX INT8:           34.6 ms/张
      INT8 加速比:            51.52x

    ============================================================
    ✓ INT8 静态量化完成: best_detection_int8_static.onnx
      模型大小: 47.6 MB → 13.5 MB (3.5x 压缩)
      用法: 直接替换推理脚本中的模型路径
      示例: python detection_inference.py --model_path best_detection_int8_static.onnx
    ============================================================
    (.venv) weiqiang@weiqiang:~/AI/ocr-majorbio$ python detection_quantize_int8_v2.py
    ============================================================
    EAST 模型 INT8 静态量化 v2
    ============================================================
      输入模型:     best_detection.onnx
      输出模型:     best_detection_int8_static.onnx
      量化格式:     QOperator (U8S8) → 文件体积缩小
      校准目录:     (随机数据)
      校准样本数:   100
      Per-channel:  False
      Reduce-range: False
      校准方法:     MinMax
      输入尺寸:     640x640

    [1/6] 原始 FP32 模型大小: 47.6 MB

    [2/6] 分析模型结构，定位敏感输出节点...
      自动排除输出头节点: ['node_conv2d_23', 'node_sigmoid', 'node_conv2d_24']
        → 这些节点保留 FP32 精度，避免 score/geo 量化误差

    [3/6] 模型预处理 (shape inference + optimization)...
      预处理模型: best_detection.onnx → best_detection_prep.onnx
      ✓ 预处理完成
      自动排除输出头节点: ['node_conv2d_23', 'node_sigmoid', 'node_conv2d_24']
        → 这些节点保留 FP32 精度，避免 score/geo 量化误差

    [4/6] 准备校准数据...
      使用随机校准数据 (100 条)

    [5/6] 执行 INT8 静态量化 (QOperator)...
      ✓ 量化完成！ (耗时 118.9s)
      FP32 模型:  47.6 MB
      INT8 模型:  13.5 MB
      压缩比:     3.51x

    [6/6] 验证 FP32 vs INT8 输出一致性...
      score_map 最大误差:   1.6132e-01
      score_map 平均误差:   3.2991e-05
      score_map 相关系数:   0.993084
      geo_map   最大误差:   4.9910e-01
      geo_map   平均误差:   5.7317e-02
      ✓ 精度优秀，量化几乎无损

    [Benchmark] 速度对比 (CPU, 5次预热 + 20次计时)...
      ONNX FP32:           1514.4 ms/张
      ONNX INT8:           35.4 ms/张
      INT8 加速比:            42.72x

    ============================================================
    ✓ INT8 静态量化完成: best_detection_int8_static.onnx
      模型大小: 47.6 MB → 13.5 MB (3.5x 压缩)
      用法: 直接替换推理脚本中的模型路径
      示例: python detection_inference.py --model_path best_detection_int8_static.onnx