package com.appforcross.core.render

import com.appforcross.core.image.Raster

interface PreviewRenderer {
        fun render(
                patternLike: Any,
                cellPx: Int,
                showGrid: Boolean,
                showSymbols: Boolean
            ): Raster
    }
