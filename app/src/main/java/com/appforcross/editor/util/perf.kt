package com.appforcross.editor.util

import android.os.SystemClock
import android.util.Log

/**
 * Утилита для измерения времени выполнения и логирования шагов.
 * Используется для профилирования тяжёлых операций (детект, пайплайны и т.д.)
 */
object Perf {

    /**
     * Замер синхронных блоков.
     * Пример: Perf.trace("TAG", "DiscretePipeline.run") { ... }
     */
    inline fun <T> trace(tag: String, label: String, block: () -> T): T {
        val t0 = SystemClock.elapsedRealtimeNanos()
        Log.i(tag, "▶ $label")
        val out = try {
            block()
        } finally {
            val dtMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000.0
            Log.i(tag, "◀ $label in ${"%.1f".format(dtMs)} ms")
        }
        return out
    }

    /**
     * Замер suspend-функций.
     * Пример: Perf.traceSuspend("TAG", "SmartScene.detect") { ... }
     */
    suspend fun <T> traceSuspend(tag: String, label: String, block: suspend () -> T): T {
        val t0 = SystemClock.elapsedRealtimeNanos()
        Log.i(tag, "▶ $label")
        val out = try {
            block()
        } finally {
            val dtMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000.0
            Log.i(tag, "◀ $label in ${"%.1f".format(dtMs)} ms")
        }
        return out
    }

    /**
     * Короткий лог по использованию памяти (полезно после тяжёлых шагов).
     * Пример: Perf.heap("AutoProcessor", "after PhotoPipeline.run")
     */
    fun heap(tag: String, label: String) {
        val rt = Runtime.getRuntime()
        val usedMb = (rt.totalMemory() - rt.freeMemory()) / (1024.0 * 1024.0)
        Log.i(tag, "$label heap=${"%.1f".format(usedMb)} MB")
    }
}
