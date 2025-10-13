package com.appforcross.core.metrics

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import androidx.annotation.CheckResult
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asAndroidBitmap

/**
 * Даунскейл изображений для быстрых метрик.
 * Все функции возвращают ARGB_8888.
 */
object Downscale {

    /** Масштабировать [src] по ширине до [target], с сохранением пропорций. */
    @CheckResult
    fun toWidth(src: Bitmap, target: Int): Bitmap {
        if (target <= 0) return src
        val w = src.width
        if (w == 0) return src
        if (w == target) return ensureArgb8888(src)
        val ratio = target.toFloat() / w
        val h = (src.height * ratio).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(ensureArgb8888(src), target, h, /*filter*/ true)
    }

    /** Масштабировать [src] так, чтобы максимальная сторона стала [maxSide]. */
    @CheckResult
    fun toMaxSide(src: Bitmap, maxSide: Int): Bitmap {
        if (maxSide <= 0) return src
        val w = src.width
        val h = src.height
        if (w == 0 || h == 0) return src
        val curMax = maxOf(w, h)
        if (curMax == maxSide) return ensureArgb8888(src)
        val ratio = maxSide.toFloat() / curMax
        val nw = (w * ratio).toInt().coerceAtLeast(1)
        val nh = (h * ratio).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(ensureArgb8888(src), nw, nh, true)
    }

    /** Перегрузки для ImageBitmap. */
    @CheckResult
    fun toWidth(src: ImageBitmap, target: Int): Bitmap = toWidth(src.asAndroidBitmap(), target)

    @CheckResult
    fun toMaxSide(src: ImageBitmap, maxSide: Int): Bitmap = toMaxSide(src.asAndroidBitmap(), maxSide)

    /** Гарантировать ARGB_8888 (для унифицированной работы пикселей/метрик). */
    @CheckResult
    private fun ensureArgb8888(src: Bitmap): Bitmap {
        if (src.config == Bitmap.Config.ARGB_8888) return src
        val out = Bitmap.createBitmap(src.width, src.height, Bitmap.Config.ARGB_8888)
        Canvas(out).drawBitmap(src, 0f, 0f, Paint(Paint.ANTI_ALIAS_FLAG))
        return out
    }
}
