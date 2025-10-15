package com.appforcross.editor.auto.photo.hq

import android.graphics.Bitmap
import android.util.Log

/**
 * Новый оркестратор: выбор масштаба → пресет → палитра → рендер → приёмка (ретрай при необходимости).
 * Вся логика вынесена в модули: ScaleSelector / PresetGate / PaletteBuilder / Renderer / AcceptanceGuard.
 */
object PhotoHQ {

    data class Result(
        val bitmap: Bitmap,
        val usedSwatches: Int,
        val S: Int,
        val params: PresetGate.Params,
        val metrics: AcceptanceGuard.Metrics
    )

    /**
     * Главная точка входа. Минимальный оркестратор, без бизнес‑логики.
     *
     * @param source         исходный Bitmap
     * @param threadPalette  палитра ниток (может быть List<Swatch>, принимаем как List<*>)
     * @param scaleCandidates список S (длинная сторона) для оценки
     */
    @JvmStatic
    fun process(
        source: Bitmap,
        threadPalette: List<*>,
        scaleCandidates: IntArray = intArrayOf(180, 200, 240, 300, 360, 400, 480, 600, 720)
    ): Result {
        // 1) Выбор масштаба + кэш превью
        val choice = ScaleSelector.select(source, scaleCandidates)
        Log.d("PhotoHQ.Scale", "pick S=${choice.S} probe=${choice.probe}")

        // 2) Пресет/ручки (Preset Gate)
        val feats = PresetGate.computeFeatures(choice.bmpS)
        val (preset, params0) = PresetGate.pick(feats)
        Log.d("PhotoHQ.Policy", "preset=$preset feats=$feats params=$params0")

        // 3) Палитра под S
        val palette0 = PaletteBuilder.build(choice.bmpS, threadPalette, params0)
        Log.d("PhotoHQ.RAQ", "palette size=${palette0.argb.size}")

        // 4) Рендер и метрики
        val frame0 = Renderer.render(choice.bmpS, palette0, params0)
        val acc0 = AcceptanceGuard.collect(frame0.logs)
        Log.d("PhotoHQ.Metrics", acc0.toString())

        // 5) Один авто‑ретрай при провале порогов
        val needRetry = AcceptanceGuard.needRetry(preset, acc0)
        val (frameF, paramsF, accF) =
            if (!needRetry) Triple(frame0, params0, acc0)
            else {
            val params1 = AcceptanceGuard.adjust(params0)
            Log.d("PhotoHQ.Policy", "retry params=$params1 cause=$acc0")
            val palette1 = PaletteBuilder.build(choice.bmpS, threadPalette, params1)
            val frame1 = Renderer.render(choice.bmpS, palette1, params1)
            val acc1 = AcceptanceGuard.collect(frame1.logs)
            Triple(frame1, params1, acc1)
        }

        return Result(
            bitmap = frameF.bitmap,
            usedSwatches = frameF.usedSwatches,
            S = choice.S,
            params = paramsF,
            metrics = accF
        )
    }

    // === Совместимость со старым кодом (мягкие шунты) =========================
    // Если где‑то ещё вызываются старые API, эти методы позволят собрать проект.
    // По мере миграции можно удалить.

    @Deprecated("Use process(...) orchestration instead")
    @JvmStatic
    fun buildPaletteForS(
        source: Bitmap,
        threadPalette: List<*>,
        K: Int,
        blur: Float,
        targetWidthSt: Int
    ): Pair<IntArray, FloatArray> {
        val scaled = ScaleSelector.scaleToMaxSide(source, targetWidthSt)
        val dummyParams = PresetGate.Params(
            kMin = K,
            minAfterMerge = K.coerceAtLeast(24),
            skinMin = 0, skyMin = 0, waterMin = 0, neutralMin = 0,
            ampBase = 0.18f, ampStrong = 0.24f, fsTarget = 0.25f
        )
        val pal = PaletteBuilder.build(scaled, threadPalette, dummyParams)
        return pal.argb to pal.lab
    }

    @Deprecated("Use process(...) orchestration instead")
    @JvmStatic
    fun renderWithFrozenPaletteAtS(
        source: Bitmap,
        threadPalette: List<*>,
        paletteArgb: IntArray,
        paletteLab: FloatArray,
        S: Int,
        K: Int,
        blur: Float
    ): Bitmap {
        val scaled = ScaleSelector.scaleToMaxSide(source, S)
        val pal = PaletteBuilder.Palette(paletteArgb, paletteLab)
        val params = PresetGate.Params(
            kMin = K, minAfterMerge = K.coerceAtLeast(24),
            skinMin = 0, skyMin = 0, waterMin = 0, neutralMin = 0,
            ampBase = 0.18f, ampStrong = 0.24f, fsTarget = 0.25f
        )
        return Renderer.render(scaled, pal, params).bitmap
    }
}

