const b=1
resnet18_layers_tiny = [

    # Initial conv layer
    (id=1, input_shape=(64, 64, 3, b), kernel_shape=(7, 7, 3, 64), stride=2, padding=3, tag=nothing),

    # Stage 1: Conv2_x (2 BasicBlocks × 2 convs each), all stride=1, channels=64
    (id=2, input_shape=(32, 32, 64, b), kernel_shape=(3, 3, 64, 64), stride=1, padding=1, tag="residual_in"),
    (id=3, input_shape=(32, 32, 64, b), kernel_shape=(3, 3, 64, 64), stride=1, padding=1, tag="residual_add"),

    (id=4, input_shape=(32, 32, 64, b), kernel_shape=(3, 3, 64, 64), stride=1, padding=1, tag="residual_in"),
    (id=5, input_shape=(32, 32, 64, b), kernel_shape=(3, 3, 64, 64), stride=1, padding=1, tag="residual_add"),

    # Stage 2: Conv3_x (2 BasicBlocks × 2 convs each), first block stride=2
    (id=6, input_shape=(32, 32, 64, b), kernel_shape=(3, 3, 64, 128), stride=2, padding=1, tag="residual_projection"),
    (id=7, input_shape=(16, 16, 128, b), kernel_shape=(3, 3, 128, 128), stride=1, padding=1, tag="residual_add"),

    (id=8, input_shape=(16, 16, 128, b), kernel_shape=(3, 3, 128, 128), stride=1, padding=1, tag="residual_in"),
    (id=9, input_shape=(16, 16, 128, b), kernel_shape=(3, 3, 128, 128), stride=1, padding=1, tag="residual_add"),

    # Stage 3: Conv4_x (2 BasicBlocks × 2 convs each), first block stride=2
    (id=10, input_shape=(16, 16, 128, b), kernel_shape=(3, 3, 128, 256), stride=2, padding=1, tag="residual_projection"),
    (id=11, input_shape=(8, 8, 256, b), kernel_shape=(3, 3, 256, 256), stride=1, padding=1, tag="residual_add"),

    (id=12, input_shape=(8, 8, 256, b), kernel_shape=(3, 3, 256, 256), stride=1, padding=1, tag="residual_in"),
    (id=13, input_shape=(8, 8, 256, b), kernel_shape=(3, 3, 256, 256), stride=1, padding=1, tag="residual_add"),

    # Stage 4: Conv5_x (2 BasicBlocks × 2 convs each), first block stride=2
    (id=14, input_shape=(8, 8, 256, b), kernel_shape=(3, 3, 256, 512), stride=2, padding=1, tag="residual_projection"),
    (id=15, input_shape=(4, 4, 512, b), kernel_shape=(3, 3, 512, 512), stride=1, padding=1, tag="residual_add"),

    (id=16, input_shape=(4, 4, 512, b), kernel_shape=(3, 3, 512, 512), stride=1, padding=1, tag="residual_in"),
    (id=17, input_shape=(4, 4, 512, b), kernel_shape=(3, 3, 512, 512), stride=1, padding=1, tag="residual_add"),

    # Average Pooling and Fully Connected layer
    (id=18, input_shape=(4, 4, 512, b), kernel_shape=(4, 4, 512, 512), stride=1, padding=0, tag="avg_pool"),
    (id=19, input_shape=(512,), output_shape=(100,), stride=1, padding=1, tag="fc")
]