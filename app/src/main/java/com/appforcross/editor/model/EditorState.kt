//Hash c83c86c3795ead10f6964717268cf15c
package com.appforcross.editor.model

import androidx.compose.ui.graphics.ImageBitmap

data class PreprocessState(
    val brightnessPct: Int = 0,      // -100..+100
    val contrastPct: Int = 0,        // -100..+100
    val gamma: Float = 1.0f,         // 0.10..3.00
    val autoLevels: Boolean = false,
    val denoise: DenoiseLevel = DenoiseLevel.NONE,
    val tonalCompression: Float = 0f // 0.0..1.0
)

data class SizeState(
    val unit: UnitMode = UnitMode.ST_PER_INCH,
    val presetDensity: Float = 14f,
    val pick: SizePick = SizePick.BY_WIDTH,
    val widthStitches: Int = 120,
    val heightStitches: Int = 120,
    val keepAspect: Boolean = true,
    val pagesTarget: Int = 1
)

data class PaletteState(
    val maxColors: Int = 0, // 0 = unlimited (after k-means)
    val metric: ColorMetric = ColorMetric.DE2000,
    val dithering: DitheringType = DitheringType.NONE
)

data class OptionsState(
    val useDmcAsIs: Boolean = true,
    val mergeDeltaE: Float = 0f,
    val fsStrengthPct: Int = 50,
    val cleanSingles: Boolean = true,
    val resampling: ResamplingMode = ResamplingMode.AVERAGE_PER_CELL,
    val preBlurSigmaPx: Float = 0f
)

data class EditorState(
    val sourceImage: ImageBitmap? = null,
    val previewImage: ImageBitmap? = null,

    val preprocess: PreprocessState = PreprocessState(),
    val size: SizeState = SizeState(),
    val palette: PaletteState = PaletteState(),
    val options: OptionsState = OptionsState(),

    val aspect: Float = 1f, // width / height
    val isBusy: Boolean = false,
    val error: String? = null
)