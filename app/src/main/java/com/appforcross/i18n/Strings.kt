
package com.appforcross.i18n

interface Strings {
    val nav: NavStrings
    val common: CommonStrings
    val import: ImportStrings
    val preview: PreviewStrings
    val palette: PaletteStrings
    val preprocess: PreprocessStrings
    val size: SizeStrings
    val options: OptionsStrings
}

interface NavStrings {
    val editorTitle: String
    val tabImport: String
    val tabPreprocess: String
    val tabSize: String
    val tabPalette: String
    val tabOptions: String
    val tabPreview: String
    }

interface CommonStrings {
    val apply: String
    val open: String
    val share: String
    val delete: String
    val ok: String
    val cancel: String
    val undo: String
    val redo: String
    }

interface ImportStrings {
    val sectionTitle: String
    val description: String
    val selectImage: String
    val miniPreview: String
    val imageNotSelected: String
    val recentExports: String
    fun sizePx(w: Int, h: Int): String
    }

interface PreviewStrings {
    val preview: String
    val grid: String
    val symbols: String
    val exportCellLabel: String
    fun exportCellPx(px: Int): String
    val exportPng: String
    val exportPdf: String
    val promptImport: String
    // zoom controls / HUD
    val zoomFit: String
    val zoomFill: String
    val zoom100: String
    val zoomIn: String
    val zoomOut: String
    fun zoomHud(percent: Int): String
}

interface PaletteStrings {
    val assetsTitle: String
    val maxColors: String
    val zeroMeansNoLimit: String
    val metricTitle: String
    val metricDE2000: String
    val metricDE76: String
    val metricOKLAB: String
    val ditheringTitle: String
    val ditheringNone: String
    val ditheringFs: String
    val ditheringAtkinson: String
    val kmeansNote: String
    val threadsTitle: String
    val estimateParams: String
    val fabricStPerInch: String
    val strands: String
    val wastePct: String
    val pressApplyToCalcThreads: String
    val setSymbols: String
    fun symbolFor(code: String): String
    val symbol1char: String
    val autoSymbols: String
}

interface PreprocessStrings {
    val sectionBrightContrast: String
    fun brightnessPct(v: Int): String
    fun contrastPct(v: Int): String
    val sectionGamma: String
    fun gammaValue(v: Float): String
    val autoLevels: String
    val sectionDenoise: String
    val denoiseNone: String
    val denoiseLow: String
    val denoiseMedium: String
    val denoiseHigh: String
    val sectionTonal: String
    fun tonalCompression(v: Float): String
    val noteAppliedBefore: String
}
interface SizeStrings {
    val title: String
    val stPerInch: String
    val stPerCm: String
    val preset: String
    val byWidth: String
    val byHeight: String
    val byDpi: String
    val widthStitches: String
    val heightStitches: String
    val density: String
    val keepAspect: String
    val pagesTarget: String
    fun densityLabelInch(d: Float): String
    fun densityLabelCm(d: Float): String
    fun physicalSummary(preset: Float, wIn: Float, wCm: Float, hIn: Float, hCm: Float): String
    fun physicalSizeText(wIn: Float, hIn: Float): String
}

interface OptionsStrings {
    val paletteTitle: String
    val dmcAsIs: String
    val adaptiveToDmc: String
    fun mergeDeltaE(v: Float): String
    val ditherTitle: String
    fun ditherStrength(pct: Int): String
    val cleanSingles: String
    val resampleTitle: String
    val avgPerCell: String
    val simpleFast: String
    fun preblurSigma(v: Float): String
    val recommendation: String
}
