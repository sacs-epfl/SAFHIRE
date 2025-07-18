const b=1

resnet20_layers_cifar = [
    # Initial conv layer
    (id=1, input_shape=(32, 32, 3, b), kernel_shape=(3, 3, 3, 16), stride=1, padding=1, tag=nothing),

    # Stage 1: Conv2_x (3 blocks × 2 conv layers, all stride=1, padding=1)
    (id=2, input_shape=(32, 32, 16, b), kernel_shape=(3, 3, 16, 16), stride=1, padding=1, tag="residual_in"),
    (id=3, input_shape=(32, 32, 16, b), kernel_shape=(3, 3, 16, 16), stride=1, padding=1, tag="residual_add"),

    (id=4, input_shape=(32, 32, 16, b), kernel_shape=(3, 3, 16, 16), stride=1, padding=1, tag="residual_in"),
    (id=5, input_shape=(32, 32, 16, b), kernel_shape=(3, 3, 16, 16), stride=1, padding=1, tag="residual_add"),

    (id=6, input_shape=(32, 32, 16, b), kernel_shape=(3, 3, 16, 16), stride=1, padding=1, tag="residual_in"),
    (id=7, input_shape=(32, 32, 16, b), kernel_shape=(3, 3, 16, 16), stride=1, padding=1, tag="residual_add"),

    # Stage 2: Conv3_x (3 blocks × 2 conv layers, first conv of first block uses stride=2)
    (id=8, input_shape=(32, 32, 16, b), kernel_shape=(3, 3, 16, 32), stride=2, padding=1, tag="residual_projection"),
    (id=9, input_shape=(16, 16, 32, b), kernel_shape=(3, 3, 32, 32), stride=1, padding=1, tag="residual_add"),

    (id=10, input_shape=(16, 16, 32, b), kernel_shape=(3, 3, 32, 32), stride=1, padding=1, tag="residual_in"),
    (id=11, input_shape=(16, 16, 32, b), kernel_shape=(3, 3, 32, 32), stride=1, padding=1, tag="residual_add"),

    (id=12, input_shape=(16, 16, 32, b), kernel_shape=(3, 3, 32, 32), stride=1, padding=1, tag="residual_in"),
    (id=13, input_shape=(16, 16, 32, b), kernel_shape=(3, 3, 32, 32), stride=1, padding=1, tag="residual_add"),

    # Stage 3: Conv4_x (3 blocks × 2 conv layers, first conv of first block uses stride=2)
    (id=14, input_shape=(16, 16, 32, b), kernel_shape=(3, 3, 32, 64), stride=2, padding=1, tag="residual_projection"),
    (id=15, input_shape=(8, 8, 64, b), kernel_shape=(3, 3, 64, 64), stride=1, padding=1, tag="residual_add"),

    (id=16, input_shape=(8, 8, 64, b), kernel_shape=(3, 3, 64, 64), stride=1, padding=1, tag="residual_in"),
    (id=17, input_shape=(8, 8, 64, b), kernel_shape=(3, 3, 64, 64), stride=1, padding=1, tag="residual_add"),

    (id=18, input_shape=(8, 8, 64, b), kernel_shape=(3, 3, 64, 64), stride=1, padding=1, tag="residual_in"),
    (id=19, input_shape=(8, 8, 64, b), kernel_shape=(3, 3, 64, 64), stride=1, padding=1, tag="residual_add"),

    # Average Pooling performed as a sum on W and H
    (id=20, input_shape=(8, 8, 64, b), kernel_shape=(8, 8, 64, 64), stride=1, padding=0, tag="avg_pool"),

    (id=21, input_shape=(64,),output_shape=(10), stride=1, padding=1, tag="fc")
]