b = 1  # placeholder for batch size variable

resnet34_layers = [
    # Initial conv layer
    (id=1, input_shape=(224, 224, 3, b), kernel_shape=(7, 7, 3, 64), stride=2, padding=3, tag=nothing),

    # MaxPool layer - replaced by average pooling
    (id=2, input_shape=(112, 112, 64, b), kernel_shape=(3, 3, 64, 64), stride=2, padding=1, tag=nothing),

    # Conv2_x block (Stage 1) - 3 blocks × 2 conv layers each (stride=1, padding=1, kernel=3)
    (id=3, input_shape=(56, 56, 64, b), kernel_shape=(3, 3, 64, 64), stride=1, padding=1, tag="residual_in"),
    (id=4, input_shape=(56, 56, 64, b), kernel_shape=(3, 3, 64, 64), stride=1, padding=1, tag="residual_add"),

    (id=5, input_shape=(56, 56, 64, b), kernel_shape=(3, 3, 64, 64), stride=1, padding=1, tag="residual_in"),
    (id=6, input_shape=(56, 56, 64, b), kernel_shape=(3, 3, 64, 64), stride=1, padding=1, tag="residual_add"),

    (id=7, input_shape=(56, 56, 64, b), kernel_shape=(3, 3, 64, 64), stride=1, padding=1, tag="residual_in"),
    (id=8, input_shape=(56, 56, 64, b), kernel_shape=(3, 3, 64, 64), stride=1, padding=1, tag="residual_add"),

    # Conv3_x block (Stage 2) - 4 blocks × 2 conv layers, first conv stride=2 downsamples
    (id=9, input_shape=(56, 56, 64, b), kernel_shape=(3, 3, 64, 128), stride=2, padding=1, tag="residual_projection"),
    (id=10, input_shape=(28, 28, 128, b), kernel_shape=(3, 3, 128, 128), stride=1, padding=1, tag="residual_add"),

    (id=11, input_shape=(28, 28, 128, b), kernel_shape=(3, 3, 128, 128), stride=1, padding=1, tag="residual_in"),
    (id=12, input_shape=(28, 28, 128, b), kernel_shape=(3, 3, 128, 128), stride=1, padding=1, tag="residual_add"),

    (id=13, input_shape=(28, 28, 128, b), kernel_shape=(3, 3, 128, 128), stride=1, padding=1, tag="residual_in"),
    (id=14, input_shape=(28, 28, 128, b), kernel_shape=(3, 3, 128, 128), stride=1, padding=1, tag="residual_add"),

    (id=15, input_shape=(28, 28, 128, b), kernel_shape=(3, 3, 128, 128), stride=1, padding=1, tag="residual_in"),
    (id=16, input_shape=(28, 28, 128, b), kernel_shape=(3, 3, 128, 128), stride=1, padding=1, tag="residual_add"),

    # Conv4_x block (Stage 3) - 6 blocks × 2 conv layers, first conv stride=2 downsamples
    (id=17, input_shape=(28, 28, 128, b), kernel_shape=(3, 3, 128, 256), stride=2, padding=1, tag="residual_projection"),
    (id=18, input_shape=(14, 14, 256, b), kernel_shape=(3, 3, 256, 256), stride=1, padding=1, tag="residual_add"),

    (id=19, input_shape=(14, 14, 256, b), kernel_shape=(3, 3, 256, 256), stride=1, padding=1, tag="residual_in"),
    (id=20, input_shape=(14, 14, 256, b), kernel_shape=(3, 3, 256, 256), stride=1, padding=1, tag="residual_add"),

    (id=21, input_shape=(14, 14, 256, b), kernel_shape=(3, 3, 256, 256), stride=1, padding=1, tag="residual_in"),
    (id=22, input_shape=(14, 14, 256, b), kernel_shape=(3, 3, 256, 256), stride=1, padding=1, tag="residual_add"),

    (id=23, input_shape=(14, 14, 256, b), kernel_shape=(3, 3, 256, 256), stride=1, padding=1, tag="residual_in"),
    (id=24, input_shape=(14, 14, 256, b), kernel_shape=(3, 3, 256, 256), stride=1, padding=1, tag="residual_add"),

    (id=25, input_shape=(14, 14, 256, b), kernel_shape=(3, 3, 256, 256), stride=1, padding=1, tag="residual_in"),
    (id=26, input_shape=(14, 14, 256, b), kernel_shape=(3, 3, 256, 256), stride=1, padding=1, tag="residual_add"),

    (id=27, input_shape=(14, 14, 256, b), kernel_shape=(3, 3, 256, 256), stride=1, padding=1, tag="residual_in"),
    (id=28, input_shape=(14, 14, 256, b), kernel_shape=(3, 3, 256, 256), stride=1, padding=1, tag="residual_add"),

    # Conv5_x block (Stage 4) - 3 blocks × 2 conv layers, first conv stride=2 downsamples
    (id=29, input_shape=(14, 14, 256, b), kernel_shape=(3, 3, 256, 512), stride=2, padding=1, tag="residual_projection"),
    (id=30, input_shape=(7, 7, 512, b), kernel_shape=(3, 3, 512, 512), stride=1, padding=1, tag="residual_add"),

    (id=31, input_shape=(7, 7, 512, b), kernel_shape=(3, 3, 512, 512), stride=1, padding=1, tag="residual_in"),
    (id=32, input_shape=(7, 7, 512, b), kernel_shape=(3, 3, 512, 512), stride=1, padding=1, tag="residual_add"),

    (id=33, input_shape=(7, 7, 512, b), kernel_shape=(3, 3, 512, 512), stride=1, padding=1, tag="residual_in"),
    (id=34, input_shape=(7, 7, 512, b), kernel_shape=(3, 3, 512, 512), stride=1, padding=1, tag="residual_add"),

        # Average Pooling performed as a sum on W and H
    (id=35, input_shape=(7, 7, 512, b), kernel_shape=(7, 7, 512, 512), stride=1, padding=0, tag="avg_pool"),

    (id=36, input_shape=(512,),output_shape=(1000,), stride=1, padding=1, tag="fc")
]