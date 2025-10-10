package com.appforcross.core.image

import kotlinx.serialization.Serializable

enum class GridMode { EXACT, BY_PAGES }
enum class PageFormat { A4, A3, LETTER }
enum class QuantAlgorithm { NEAREST, FLOYD_STEINBERG, TOP_N }
enum class DitheringStrength { NONE, LOW, MEDIUM, HIGH }

@Serializable
data class GridSettings(
    val mode: GridMode = GridMode.EXACT,
    val stitchesW: Int = 200,
    val stitchesH: Int = 130,
    val dpi: Int = 300,
    val pageFormat: PageFormat = PageFormat.A4,
    val marginMm: Int = 10,
    val desiredPages: Int = 1
)

@Serializable
data class OptionsSettings(
    val quantAlgorithm: QuantAlgorithm = QuantAlgorithm.NEAREST,
    val fsDither: Boolean = false,
    val topN: Int = 1,
    val gridBoldStep: Int = 10,
    val gridColor: String = "#808080",
    val symbolSet: String = "A1",
    val symbolSizeSp: Int = 12
)

@Serializable
data class PreprocessSettings(
    val brightness: Int = 0,
    val contrast: Int = 0,
    val gamma: Float = 1.0f,
    val denoise: Int = 0
)

@Serializable
data class AdvancedSettings(
    val strength: Int = 50,
    val smoothing: Int = 0,
    val ditheringStrength: DitheringStrength = DitheringStrength.NONE,
    val topN: Int = 1,
    val overlayAlpha: Int = 80,
    val gammaQuant: Float = 1.0f
)

@Serializable
data class ProjectParams(
    val name: String = "Без имени",
    val paletteId: String = "dmc",
    val maxColors: Int = 32,
    val grid: GridSettings = GridSettings(),
    val options: OptionsSettings = OptionsSettings(),
    val preprocess: PreprocessSettings = PreprocessSettings(),
    val advanced: AdvancedSettings = AdvancedSettings()
)

@Serializable
data class Project(
    val id: String,
    val params: ProjectParams = ProjectParams(),
    val originalImageUri: String? = null,
    val createdAt: Long = System.currentTimeMillis(),
    val updatedAt: Long = System.currentTimeMillis()
)
