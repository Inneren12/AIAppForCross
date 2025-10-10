
package com.appforcross.i18n

import android.content.res.Resources
import com.appforcross.app.R

class AndroidStrings(private val res: Resources) : Strings {
    override val nav = object : NavStrings {
        override val editorTitle = res.getString(R.string.nav_editor)
        override val tabImport = res.getString(R.string.nav_tab_import)
        override val tabPreprocess = res.getString(R.string.nav_tab_preprocess)
        override val tabSize = res.getString(R.string.nav_tab_size)
        override val tabPalette = res.getString(R.string.nav_tab_palette)
        override val tabOptions = res.getString(R.string.nav_tab_options)
        override val tabPreview = res.getString(R.string.nav_tab_preview)
    }

        override val common = object : CommonStrings {
            override val apply = res.getString(R.string.common_apply)
            override val undo = res.getString(R.string.common_undo)
            override val redo = res.getString(R.string.common_redo)
        override val open = res.getString(R.string.common_open)
        override val share = res.getString(R.string.common_share)
        override val delete = res.getString(R.string.common_delete)
        override val ok = res.getString(R.string.common_ok)
        override val cancel = res.getString(R.string.common_cancel)
    }
    override val import = object : ImportStrings {
        override val sectionTitle = res.getString(R.string.import_title)
        override val description = res.getString(R.string.import_description)
        override val selectImage = res.getString(R.string.import_select_image)
        override val miniPreview = res.getString(R.string.import_mini_preview)
        override val imageNotSelected = res.getString(R.string.import_no_image)
        override val recentExports = res.getString(R.string.import_recent_exports)
        override fun sizePx(w: Int, h: Int) = res.getString(R.string.import_size_px, w, h)
    }
    override val preview = object : PreviewStrings {
        override val preview = res.getString(R.string.preview_title)
        override val grid = res.getString(R.string.preview_grid)
        override val symbols = res.getString(R.string.preview_symbols)
        override val exportCellLabel = res.getString(R.string.preview_export_cell_label)
        override fun exportCellPx(px: Int) = res.getString(R.string.preview_export_cell_px, px)
        override val exportPng = res.getString(R.string.preview_export_png)
        override val exportPdf = res.getString(R.string.preview_export_pdf)
        override val promptImport = res.getString(R.string.preview_prompt_import)
        override val zoomFit = res.getString(R.string.preview_zoom_fit)
        override val zoom100 = res.getString(R.string.preview_zoom_100)
        override val zoomIn = res.getString(R.string.preview_zoom_in)
        override val zoomOut = res.getString(R.string.preview_zoom_out)
        override fun zoomHud(percent: Int) = res.getString(R.string.preview_zoom_hud, percent)
    }
    override val palette = object : PaletteStrings {
        override val assetsTitle = res.getString(R.string.palette_assets_title)
        override val maxColors = res.getString(R.string.palette_max_colors)
        override val zeroMeansNoLimit = res.getString(R.string.palette_zero_means_no_limit)
        override val metricTitle = res.getString(R.string.palette_metric_title)
        override val metricDE2000 = res.getString(R.string.palette_metric_de2000)
        override val metricDE76 = res.getString(R.string.palette_metric_de76)
        override val metricOKLAB = res.getString(R.string.palette_metric_oklab)
        override val ditheringTitle = res.getString(R.string.palette_dithering_title)
        override val ditheringNone = res.getString(R.string.palette_dithering_none)
        override val ditheringFs = res.getString(R.string.palette_dithering_fs)
        override val ditheringAtkinson = res.getString(R.string.palette_dithering_atkinson)
        override val kmeansNote = res.getString(R.string.palette_kmeans_note)
        override val threadsTitle = res.getString(R.string.palette_threads_title)
        override val estimateParams = res.getString(R.string.palette_estimate_params)
        override val fabricStPerInch = res.getString(R.string.palette_fabric_st_per_inch)
        override val strands = res.getString(R.string.palette_strands)
        override val wastePct = res.getString(R.string.palette_waste_pct)
        override val pressApplyToCalcThreads = res.getString(R.string.palette_press_apply_to_calc_threads)
        override val setSymbols = res.getString(R.string.palette_set_symbols)
        override fun symbolFor(code: String) = res.getString(R.string.palette_symbol_for_x, code)
        override val symbol1char = res.getString(R.string.palette_symbol_1_char)
        override val autoSymbols = res.getString(R.string.palette_auto_symbols)
    }
    override val preprocess = object : PreprocessStrings {
        override val sectionBrightContrast = res.getString(R.string.preprocess_bright_contrast)
        override fun brightnessPct(v: Int) = res.getString(R.string.preprocess_brightness_pct, v)
        override fun contrastPct(v: Int) = res.getString(R.string.preprocess_contrast_pct, v)
        override val sectionGamma = res.getString(R.string.preprocess_gamma)
        override fun gammaValue(v: Float) = res.getString(R.string.preprocess_gamma_value, v)
        override val autoLevels = res.getString(R.string.preprocess_auto_levels)
        override val sectionDenoise = res.getString(R.string.preprocess_denoise)
        override val denoiseNone = res.getString(R.string.preprocess_denoise_none)
        override val denoiseLow = res.getString(R.string.preprocess_denoise_low)
        override val denoiseMedium = res.getString(R.string.preprocess_denoise_medium)
        override val denoiseHigh = res.getString(R.string.preprocess_denoise_high)
        override val sectionTonal = res.getString(R.string.preprocess_tonal)
        override fun tonalCompression(v: Float) = res.getString(R.string.preprocess_tonal_value, v)
        override val noteAppliedBefore = res.getString(R.string.preprocess_note_before)
    }
    override val size = object : SizeStrings {
        override val title = res.getString(R.string.size_title)
        override val stPerInch = res.getString(R.string.size_st_per_inch)
        override val stPerCm = res.getString(R.string.size_st_per_cm)
        override val preset = res.getString(R.string.size_preset)
        override val byWidth = res.getString(R.string.size_by_width)
        override val byHeight = res.getString(R.string.size_by_height)
        override val byDpi = res.getString(R.string.size_by_dpi)
        override val widthStitches = res.getString(R.string.size_width_stitches)
        override val heightStitches = res.getString(R.string.size_height_stitches)
        override val density = res.getString(R.string.size_density)
        override val keepAspect = res.getString(R.string.size_keep_aspect)
        override val pagesTarget = res.getString(R.string.size_pages_target)
        override fun densityLabelInch(d: Float) = res.getString(R.string.size_density_label_inch, d.toInt())
        override fun densityLabelCm(d: Float) = res.getString(R.string.size_density_label_cm, d)
        override fun physicalSummary(preset: Float, wIn: Float, wCm: Float, hIn: Float, hCm: Float) =
            res.getString(R.string.size_physical_summary, preset, wIn, wCm, hIn, hCm)
        override fun physicalSizeText(wIn: Float, hIn: Float) =
            res.getString(R.string.size_physical_size, wIn, hIn)
    }
    override val options = object : OptionsStrings {
        override val paletteTitle = res.getString(R.string.options_palette_title)
        override val dmcAsIs = res.getString(R.string.options_dmc_as_is)
        override val adaptiveToDmc = res.getString(R.string.options_adaptive_to_dmc)
        override fun mergeDeltaE(v: Float) = res.getString(R.string.options_merge_deltae, v)
        override val ditherTitle = res.getString(R.string.options_dither_title)
        override fun ditherStrength(pct: Int) = res.getString(R.string.options_dither_strength_pct, pct)
        override val cleanSingles = res.getString(R.string.options_clean_singles)
        override val resampleTitle = res.getString(R.string.options_resample_title)
        override val avgPerCell = res.getString(R.string.options_resample_avg)
        override val simpleFast = res.getString(R.string.options_resample_simple)
        override fun preblurSigma(v: Float) = res.getString(R.string.options_preblur_sigma, v)
        override val recommendation = res.getString(R.string.options_recommendation)
    }
}
