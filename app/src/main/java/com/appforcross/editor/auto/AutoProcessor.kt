// app/src/main/java/com/appforcross/editor/auto/AutoProcessor.kt
package com.appforcross.editor.auto

import android.graphics.Bitmap
import android.util.Log
import com.appforcross.core.palette.Swatch
import com.appforcross.editor.auto.detect.SmartSceneDetector
import com.appforcross.editor.auto.discrete.DiscretePipeline
import com.appforcross.editor.auto.photo.PhotoConfig
import com.appforcross.editor.auto.photo.hq.PhotoHQ

/**
 * Тонкий use‑case без UI‑зависимостей:
 *  - решает DISCRETE/PHOTO (или принимает предварительное решение),
 *  - запускает нужный конвейер,
 *  - отдаёт компактный результат.
 */
class AutoProcessor {

    data class Output(
        val image: Bitmap,                                   // итог для предпросмотра/экспорта
        val kind: SmartSceneDetector.Mode,                   // DISCRETE | PHOTO
        val confidence: Float,                               // уверенность SmartScene
        val scores: Map<SmartSceneDetector.Mode, Float>,     // softmax по двум веткам
        val toggles: SmartSceneDetector.Toggles?,            // только для DISCRETE
        val widthStitches: Int,                              // подобранная ширина S (в "стежках")
        val usedSwatches: List<Swatch>                       // фактически задействованные нитки
    )

    /**
     * @param preDetected - можно передать уже посчитанное решение, чтобы не детектить дважды.
     */
    fun process(
        source: Bitmap,
        palette: List<Swatch>,
        preDetected: SmartSceneDetector.Decision? = null
    ): Output {
        val decision = preDetected ?: SmartSceneDetector.detect(source) // DISCRETE/PHOTO + toggles + features
        Log.d("AutoProcessor", "mode=${decision.mode} conf=${"%.2f".format(decision.confidence)} toggles=${decision.toggles}")

        return when (decision.mode) {
            SmartSceneDetector.Mode.DISCRETE -> {
                // DiscretePipeline.run(...) принимает toggles; возвращает bitmap и сеточные размеры S
                val res = DiscretePipeline.run(
                    source = source,
                    palette = palette,
                    toggles = decision.toggles ?: SmartSceneDetector.Toggles(text=false, pixelGrid=false, aa=false)
                )
                Output(
                    image = res.image,
                    kind = SmartSceneDetector.Mode.DISCRETE,
                    confidence = decision.confidence,
                    scores = decision.scores,
                    toggles = decision.toggles,
                    widthStitches = res.gridWidth,            // S (ширина в «стежках»)
                    usedSwatches = res.usedSwatches
                )
            }

            SmartSceneDetector.Mode.PHOTO -> {
                val enableDescreen = decision.features.halftone >= SmartSceneDetector.Options().halftoneGate
                val t0 = android.os.SystemClock.uptimeMillis()
                android.util.Log.i("AutoProcessor", "PHOTO2: start Orchestrator.runAuto, palette=${palette.size}, descreen=$enableDescreen")
                val res = try {
                    PhotoHQ.process(
                        source = source,
                        threadPalette = palette,
                        scaleCandidates = PhotoConfig.defaultSizes
                    )
                } catch (t: Throwable) {
                        android.util.Log.e("AutoProcessor", "PHOTO2: orchestrator failed after ${android.os.SystemClock.uptimeMillis()-t0}ms", t)
                    throw t
                    }
                android.util.Log.i("AutoProcessor", "PHOTO2: done in ${android.os.SystemClock.uptimeMillis()-t0}ms, grid=${res.gridWidth}, used=${res.usedSwatches.size}")
                Output(
                    image = res.image,
                    kind = SmartSceneDetector.Mode.PHOTO,
                    confidence = decision.confidence,
                    scores = decision.scores,
                    toggles = null,
                    widthStitches = res.gridWidth,            // ширина S, подобранная Orchestrator
                    usedSwatches = res.usedSwatches
                )
            }
        }
    }
}
