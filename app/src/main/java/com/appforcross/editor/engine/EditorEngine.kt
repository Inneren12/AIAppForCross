package com.appforcross.editor.engine

import com.appforcross.editor.model.*
import androidx.compose.ui.graphics.ImageBitmap

interface EditorEngine {
    fun applyPreprocess(src: ImageBitmap, pp: PreprocessState): ImageBitmap
    fun applySize(src: ImageBitmap, size: SizeState): ImageBitmap

    fun kMeansQuantize(
        src: ImageBitmap,
        maxColors: Int,
        metric: ColorMetric,
        dithering: DitheringType
    ): ImageBitmap

    fun applyOptions(
        src: ImageBitmap,
        opt: OptionsState,
        metric: ColorMetric
    ): ImageBitmap
}