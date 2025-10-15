// Hash d75cf96f8b76123e8ad895d49192b7e5
package com.appforcross.editor.ui.tabs

import androidx.compose.foundation.horizontalScroll
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.appforcross.editor.EditorViewModel
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.getValue
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.layout.ContentScale
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.ui.graphics.asAndroidBitmap
import android.graphics.Bitmap
import android.graphics.Canvas as AndroidCanvas
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.pdf.PdfDocument
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.max
import androidx.compose.material3.FilterChip
import androidx.compose.foundation.rememberScrollState
import android.content.Intent
import com.appforcross.i18n.LocalStrings
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.TransformOrigin
import kotlin.math.abs
import kotlin.math.roundToInt
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.ui.input.pointer.pointerInput

// Локальный тип для режима вписывания
private enum class FitMode { Fit, OneToOne, Custom }

@Composable
fun PreviewTab(vm: EditorViewModel) {
    val st by vm.state.collectAsState()
    val S = LocalStrings.current
    var showGrid by remember { mutableStateOf(false) }
    var showSymbols by remember { mutableStateOf(false) }
    val symbols by vm.symbolsPreview.collectAsState()
    // Реактивно следим за активной палитрой и перечитаем swatches при её смене
    val activePalId by vm.activePaletteId.collectAsState()
    val palSwatches = remember(activePalId) { vm.getActivePaletteSwatches() }
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    var exportCell by rememberSaveable { mutableStateOf(16) } // размер клетки при экспорте
    val cellOptions = listOf(8, 12, 16, 24)
    // История
    val canUndo by vm.canUndo.collectAsState()
    val canRedo by vm.canRedo.collectAsState()
    // --- Состояние зума/пана (только для предпросмотра; не в EditorState) ---
    var scale by rememberSaveable { mutableStateOf(1f) }       // 1f == "вписать"
    var offset by remember { mutableStateOf(Offset.Zero) }     // панорамирование
    var fitMode by remember { mutableStateOf(FitMode.Fit) }
    val minScale = 0.25f
    val maxScale = 8f

    Column(Modifier.fillMaxSize()) {
        // Локальные переключатели (не меняют глобальный стейт)
        Row(
            Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(16.dp),
        verticalAlignment = Alignment.CenterVertically
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Switch(checked = showGrid, onCheckedChange = { showGrid = it })
                Spacer(Modifier.width(8.dp)); Text(S.preview.grid)
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Switch(checked = showSymbols, onCheckedChange = { showSymbols = it })
                Spacer(Modifier.width(8.dp)); Text(S.preview.symbols)
            }
        }
            Row(
                Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 4.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
            ) {
                Text(S.preview.exportCellLabel)
            Row(
                Modifier.horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                cellOptions.forEach { opt ->
                    FilterChip(
                        selected = exportCell == opt,
                        onClick = { exportCell = opt },
                        label = { Text(S.preview.exportCellPx(opt)) }
                    )
                }
            }
            }

        BoxWithConstraints(
            Modifier
                .fillMaxWidth()
                .weight(1f),
            contentAlignment = Alignment.Center
        ) {
            val bmp = st.previewImage ?: st.sourceImage
            if (bmp != null) {
                val boxW = constraints.maxWidth
                val boxH = constraints.maxHeight
                val imgW = bmp.width
                val imgH = bmp.height
                // Базовый масштаб для "вписать"
                val fitScale = minOf(boxW.toFloat() / imgW, boxH.toFloat() / imgH)
                // Габариты слоя без дополнительного зума (масштаб = 1f)
                val drawW = (imgW * fitScale).toInt()
                val drawH = (imgH * fitScale).toInt()
                // Пересброс в "вписать" при смене изображения
                LaunchedEffect(bmp) {
                    scale = 1f
                    offset = Offset.Zero
                    fitMode = FitMode.Fit
                }
                val density = LocalDensity.current
                val drawWd = with(density) { drawW.toDp() }
                val drawHd = with(density) { drawH.toDp() }
                // Хелпер ограничений для пана
                fun clampOffset(off: Offset, sc: Float): Offset {
                    val contentW = drawW * sc
                    val contentH = drawH * sc
                    // Если контент меньше окна – центрируем и панорамирование по этой оси отключаем
                    val x = if (contentW <= drawW) {
                        (drawW - contentW) / 2f
                    } else {
                            val minX = drawW - contentW   // отрицательное
                        off.x.coerceIn(minX, 0f)
                        }
                    val y = if (contentH <= drawH) {
                        (drawH - contentH) / 2f
                    } else {
                            val minY = drawH - contentH   // отрицательное
                        off.y.coerceIn(minY, 0f)
                        }
                    return Offset(x, y)
                }
                // Кнопки управления зумом
                fun setFit() {
                    scale = 1f; offset = Offset.Zero; fitMode = FitMode.Fit
                }
                fun setOneToOne() {
                    val s = (1f / fitScale).coerceIn(minScale, maxScale)
                    scale = s
                    offset = clampOffset(offset, s)
                    fitMode = FitMode.OneToOne
                }
                fun zoomBy(f: Float, around: Offset? = null) {
                    val old = scale
                    val ns = (scale * f).coerceIn(minScale, maxScale)
                    val pivot = around ?: Offset(drawW / 2f, drawH / 2f)
                    // сохраняем позицию точки pivot
                    val newOff = (offset + (pivot / old)) - (pivot / ns)
                    scale = ns
                    offset = clampOffset(newOff, ns)
                    fitMode = FitMode.Custom
                }
                // Визуальный слой предпросмотра со всеми оверлеями — трансформируем целиком
                Box(
                    Modifier
                        .size(drawWd, drawHd)
                    .graphicsLayer {
                    transformOrigin = TransformOrigin(0f, 0f)
                    translationX = offset.x
                    translationY = offset.y
                    scaleX = scale
                    scaleY = scale
                }
                    // жесты: pinch‑zoom + pan
                .pointerInput(bmp, drawW, drawH, fitScale) {
                    detectTransformGestures(panZoomLock = true) { centroid, pan, gestureZoom, _ ->
                        val old = scale
                        val ns = (old * gestureZoom).coerceIn(minScale, maxScale)
                        // сохраняем точку под пальцами
                        val newOff = (offset + (centroid / old)) - (centroid / ns) + pan
                        scale = ns
                        offset = clampOffset(newOff, ns)
                        fitMode = if (abs(ns - 1f) < 0.01f) FitMode.Fit else FitMode.Custom
                    }
                }
                    // двойной тап: Fit <-> 100%
                .pointerInput(bmp, fitScale) {
                    detectTapGestures(onDoubleTap = { pos ->
                        val oneToOne = (1f / fitScale).coerceIn(minScale, maxScale)
                        val target = if (abs(scale - 1f) < 0.01f) oneToOne else 1f
                        val newOff = (offset + (pos / scale)) - (pos / target)
                        scale = target
                        offset = clampOffset(newOff, target)
                        fitMode = if (target == 1f) FitMode.Fit else FitMode.OneToOne
                    })
                }
                ) {
                    Image(
                        bitmap = bmp,
                        contentDescription = S.preview.preview,
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.FillBounds
                    )
                    // --- Подготовка буферов/кэшей для оверлеев ---
                    val pxBuf = remember(bmp) {
                        val ab = bmp.asAndroidBitmap()
                        val n = ab.width * ab.height
                        IntArray(n).also { ab.getPixels(it, 0, ab.width, 0, 0, ab.width, ab.height) }
                    }
                    // КЭШ: уникальные ARGB -> индекс ближайшего swatch (ускорение O(K×N) вместо O(W×H×N))
                    val colorToSwatchIndex = remember(bmp, activePalId) {
                        if (palSwatches.isEmpty()) null else {
                            val n = palSwatches.size
                            val swR = IntArray(n); val swG = IntArray(n); val swB = IntArray(n)
                            for (i in 0 until n) {
                                val a = palSwatches[i].argb
                                swR[i] = (a shr 16) and 0xFF; swG[i] = (a shr 8) and 0xFF; swB[i] = a and 0xFF
                            }
                            val uniq = LinkedHashSet<Int>(pxBuf.size / 4 + 1)
                            for (c in pxBuf) uniq.add(c or (0xFF shl 24))
                            val map = HashMap<Int, Int>(uniq.size * 2)
                            for (c in uniq) {
                                val r = (c shr 16) and 0xFF
                                val g = (c shr 8) and 0xFF
                                val b = c and 0xFF
                                var best = 0; var bestD = Int.MAX_VALUE
                                var i = 0
                                while (i < n) {
                                    val dr = r - swR[i]; val dg = g - swG[i]; val db = b - swB[i]
                                    val d = dr*dr + dg*dg + db*db
                                    if (d < bestD) { bestD = d; best = i }
                                    i++
                                }
                                map[c] = best
                            }
                            map
                        }
                    }
                    // --- Оверлеи ---
                    if (showGrid || showSymbols) {
                        Canvas(Modifier.fillMaxSize()) {
                            val cell = size.width / imgW // базовый (до зума)
                            val cellOnScreen = cell * scale
                            // Сетка 10x10
                            if (showGrid) {
                                val thin = max(1f, cell * 0.08f)
                                val thick = max(2f, cell * 0.18f)
                                for (x in 0..imgW) {
                                    val xx = x * cell
                                    val w = if (x % 10 == 0) thick else thin
                                    drawLine(
                                        color = Color.Black.copy(alpha = 0.35f),
                                        start = Offset(xx, 0f),
                                        end = Offset(xx, size.height),
                                        strokeWidth = w
                                    )
                                }
                                for (y in 0..imgH) {
                                    val yy = y * cell
                                    val w = if (y % 10 == 0) thick else thin
                                    drawLine(
                                        color = Color.Black.copy(alpha = 0.35f),
                                        start = Offset(0f, yy),
                                        end = Offset(size.width, yy),
                                        strokeWidth = w
                                    )
                                }
                            }
                            // Символы (если клетка достаточно крупная)
                            if (showSymbols && cellOnScreen >= 10f && palSwatches.isNotEmpty() && symbols.isNotEmpty()) {
                                val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                                    textAlign = Paint.Align.CENTER
                                    textSize = cell * 0.82f
                                }
                                val native = drawContext.canvas.nativeCanvas
                                val strokeW = max(1f, cell * 0.08f)
                                var idx = 0
                                for (y in 0 until imgH) {
                                    val cy = (y + 0.7f) * cell
                                    for (x in 0 until imgW) {
                                        val c = pxBuf[idx++]
                                        val best = colorToSwatchIndex?.get(c or (0xFF shl 24))
                                            ?: 0 // безопасный фоллбек
                                        val code = palSwatches[best].code
                                        val ch = symbols[code] ?: continue
                                        // Контраст символа к фону
                                        val r = (c shr 16) and 0xFF
                                        val g = (c shr 8) and 0xFF
                                        val b = c and 0xFF
                                        val lum = 0.2126f*r + 0.7152f*g + 0.0722f*b
                                        val fill = if (lum > 140) android.graphics.Color.BLACK else android.graphics.Color.WHITE
                                        val stroke = if (fill == android.graphics.Color.BLACK) android.graphics.Color.WHITE else android.graphics.Color.BLACK
                                        val cx = (x + 0.5f) * cell
                                        // Обводка
                                        paint.style = Paint.Style.STROKE
                                        paint.strokeWidth = strokeW
                                        paint.color = stroke
                                        native.drawText(ch.toString(), cx, cy, paint)
                                        // Заливка
                                        paint.style = Paint.Style.FILL
                                        paint.color = fill
                                        native.drawText(ch.toString(), cx, cy, paint)
                                    }
                                }
                            }
                        }
                    }
                }
                // --- Оверлеи управления зумом/HUD (поверх, в границах окна предпросмотра) ---
                Box(Modifier.fillMaxSize()) {
                    // Кнопки зума (справа сверху)
                    Row(
                        Modifier
                            .align(Alignment.TopEnd)
                            .padding(12.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        OutlinedButton(onClick = { zoomBy(0.8f) }) { Text(S.preview.zoomOut) }
                        OutlinedButton(onClick = { zoomBy(1.25f) }) { Text(S.preview.zoomIn) }
                        OutlinedButton(onClick = { setFit() }) { Text(S.preview.zoomFit) }
                        OutlinedButton(onClick = { setOneToOne() }) { Text(S.preview.zoom100) }
                    }
                    // HUD масштаба (справа снизу)
                    Surface(
                        shape = MaterialTheme.shapes.small,
                        tonalElevation = 2.dp,
                        modifier = Modifier
                            .align(Alignment.BottomEnd)
                            .padding(12.dp)
                    ) {
                        val pct = (scale * fitScale * 100f).roundToInt().coerceAtLeast(1)
                        Text(
                            text = S.preview.zoomHud(pct),
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                            style = MaterialTheme.typography.labelMedium
                        )
                    }
                }
            } else {
                Text(S.preview.promptImport)
                }
        }

        // --- История: Undo / Redo (расположено над экспортом) ---
        Row(
            Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalAlignment = Alignment.CenterVertically
        ) {
            OutlinedButton(
            enabled = canUndo,
            onClick = { vm.undo() }
        ) { Text(S.common.undo) }
        OutlinedButton(
            enabled = canRedo,
            onClick = { vm.redo() }
        ) { Text(S.common.redo) }
        }

        // Экспорт — «Сохранить как…» (SAF)
        // Лаунчеры системного диалога «Создать документ»
        val pngSaveLauncher = rememberLauncherForActivityResult(
            contract = ActivityResultContracts.CreateDocument("image/png")
        ) { uri ->
            val bmpCur = st.previewImage ?: st.sourceImage
            if (uri != null && bmpCur != null) {
                scope.launch(Dispatchers.Default) {
                    val out = buildExportBitmap(
                        src = bmpCur,
                        withSymbols = showSymbols,
                        withGrid = showGrid,
                        addLegend = showSymbols,
                        vm = vm,
                        scale = exportCell
                    )
                    withContext(Dispatchers.IO) {
                        ctx.contentResolver.openOutputStream(uri)?.use { os ->
                            out.compress(Bitmap.CompressFormat.PNG, 100, os)
                        }
                    }
                    withContext(Dispatchers.Main) { vm.registerExport(uri) }
                }
            }
        }
        val pdfSaveLauncher = rememberLauncherForActivityResult(
            contract = ActivityResultContracts.CreateDocument("application/pdf")
        ) { uri ->
            val bmpCur = st.previewImage ?: st.sourceImage
            if (uri != null && bmpCur != null) {
                scope.launch(Dispatchers.Default) {
                    val img = buildExportBitmap(
                        src = bmpCur,
                        withSymbols = showSymbols,
                        withGrid = showGrid,
                        addLegend = showSymbols,
                        vm = vm,
                        scale = exportCell
                    )
                    withContext(Dispatchers.IO) {
                        val pdf = PdfDocument()
                        val pageInfo = PdfDocument.PageInfo.Builder(img.width, img.height, 1).create()
                        val page = pdf.startPage(pageInfo)
                        page.canvas.drawBitmap(img, 0f, 0f, null)
                        pdf.finishPage(page)
                        ctx.contentResolver.openOutputStream(uri)?.use { os -> pdf.writeTo(os) }
                        pdf.close()
                    }
                    withContext(Dispatchers.Main) { vm.registerExport(uri) }
                }
            }
        }
        Row(
            Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            val bmp = st.previewImage ?: st.sourceImage
            val suffix = buildString {
                append("export_k${st.palette.maxColors}")
                if (showSymbols) append("_sym")
                if (showGrid) append("_grid")
                append("_cell$exportCell")
            }
            Button(
                enabled = bmp != null,
                onClick = { pngSaveLauncher.launch("$suffix.png") }
            ) { Text(S.preview.exportPng) }
            Button(
                enabled = bmp != null,
                onClick = { pdfSaveLauncher.launch("$suffix.pdf") }
            ) { Text(S.preview.exportPdf) }
        }
        // --- Пост-действия после экспорта: Открыть / Поделиться / Удалить ---
        val last by vm.lastExportUri.collectAsState()
        if (last != null) {
            Row(
                Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically
            ) {
                Button(onClick = {
                    val mime = ctx.contentResolver.getType(last!!) ?: "*/*"
                    val intent = Intent(Intent.ACTION_VIEW)
                        .setDataAndType(last, mime)
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    runCatching { ctx.startActivity(intent) }
                }) { Text(S.common.open) }
                Button(onClick = {
                    val mime = ctx.contentResolver.getType(last!!) ?: "*/*"
                    val intent = Intent(Intent.ACTION_SEND)
                        .setType(mime)
                        .putExtra(Intent.EXTRA_STREAM, last)
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    runCatching { ctx.startActivity(Intent.createChooser(intent, S.common.share)) }
                }) { Text(S.common.share) }
                OutlinedButton(onClick = {
                    if (vm.deleteExport(ctx, last!!)) {
                    // всё обновится через StateFlow; UI сам спрячется
                    }
                }) { Text(S.common.delete) }
            }
        }
    }
}

// Собирает bitmap для экспорта с учётом символов/сетки/легенды
private fun buildExportBitmap(
    src: androidx.compose.ui.graphics.ImageBitmap,
    withSymbols: Boolean,
    withGrid: Boolean,
    addLegend: Boolean,
    vm: EditorViewModel,
    scale: Int = 16
): Bitmap {
    val ab = src.asAndroidBitmap()
    val w = ab.width
    val h = ab.height
    val cell = scale.toFloat()
    val outW = (w * cell).toInt()
    val outHimg = (h * cell).toInt()
    // Легенда (если символы есть)
    val symbols = vm.symbolsPreview.value
    val threads = vm.threads.value
    //  Предрасчёт легенды (flow‑layout по ширине outW)
    val wantLegend = withSymbols && addLegend && symbols.isNotEmpty() && threads.isNotEmpty()
    val margin = max(12f, cell * 0.75f)
    val hGap = max(8f, cell * 0.75f)
    val vGap = max(6f, cell * 0.5f)
    val swatch = max(16f, cell * 1.2f)
    val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    paint.textAlign = Paint.Align.LEFT
    paint.textSize = max(16f, cell * 1.2f) // при cell=16 ≈ 19.2f
    val base = (paint.fontMetrics.descent - paint.fontMetrics.ascent).coerceAtLeast(16f)
    val rowH = (max(swatch, base) + vGap).toInt()
    var curX = margin
    var curY = margin
    val layoutItems = mutableListOf<LegendItem>() // для второй фазы рисования
    if (wantLegend) {
        threads.forEach { t ->
            val ch = symbols[t.code] ?: return@forEach
            val symW = paint.measureText(ch.toString())
            val codeW = paint.measureText(t.code)
            val pct = "${t.percent}%"
            val pctW = paint.measureText(pct)
            val nameW = paint.measureText(t.name)
            val itemW = swatch + hGap + symW + hGap + codeW + hGap + nameW + hGap + pctW
            if (curX + itemW > outW - margin) {
                // перенос строки
                curX = margin
                curY += rowH
            }
            layoutItems += LegendItem(
                x = curX, y = curY.toFloat(),
                swatch = t.argb, symbol = ch, code = t.code, name = t.name, percent = pct,
                widths = floatArrayOf(symW, codeW, nameW, pctW)
            )
            curX += itemW + hGap
        }
    }
    val legendH = if (wantLegend) (curY + rowH + margin - margin).toInt() else 0
    val out = Bitmap.createBitmap(outW, outHimg + legendH, Bitmap.Config.ARGB_8888)
    val cv = AndroidCanvas(out)
    // фон
    cv.drawColor(android.graphics.Color.WHITE)
    // картинка
    val dst = Rect(0, 0, outW, outHimg)
    cv.drawBitmap(ab, null, dst, null)
        // сетка
    if (withGrid) {
        paint.color = android.graphics.Color.argb(90, 0, 0, 0)
        val thin = max(1f, cell * 0.08f)
        val thick = max(2f, cell * 0.18f)
        for (x in 0..w) {
            paint.strokeWidth = if (x % 10 == 0) thick else thin
            val xx = x * cell
            cv.drawLine(xx, 0f, xx, outHimg.toFloat(), paint)
        }
        for (y in 0..h) {
            paint.strokeWidth = if (y % 10 == 0) thick else thin
            val yy = y * cell
            cv.drawLine(0f, yy, outW.toFloat(), yy, paint)
        }
    }
    // символы
    if (withSymbols && symbols.isNotEmpty()) {
        val pal = vm.getActivePaletteSwatches()
        if (pal.isNotEmpty()) {
            val n = pal.size
            val swR = IntArray(n); val swG = IntArray(n); val swB = IntArray(n)
            for (i in 0 until n) {
                val a = pal[i].argb
                swR[i] = (a shr 16) and 0xFF
                swG[i] = (a shr 8) and 0xFF
                swB[i] = a and 0xFF
            }
            val px = IntArray(w * h)
            ab.getPixels(px, 0, w, 0, 0, w, h)
            paint.textAlign = Paint.Align.CENTER
            paint.textSize = cell * 0.82f
            val strokeW = max(1f, cell * 0.08f)
            var idx = 0
            for (y in 0 until h) {
                val cy = y * cell + (cell * 0.7f)
                for (x in 0 until w) {
                    val c = px[idx++]
                    val r = (c shr 16) and 0xFF
                    val g = (c shr 8) and 0xFF
                    val b = c and 0xFF
                    var best = 0; var bestD = Int.MAX_VALUE
                    var i = 0
                    while (i < n) {
                        val dr = r - swR[i]; val dg = g - swG[i]; val db = b - swB[i]
                        val d = dr*dr + dg*dg + db*db
                        if (d < bestD) { bestD = d; best = i }
                        i++
                    }
                    val code = pal[best].code
                    val ch = symbols[code] ?: continue
                    val lum = 0.2126f*r + 0.7152f*g + 0.0722f*b
                    val fill = if (lum > 140) android.graphics.Color.BLACK else android.graphics.Color.WHITE
                    val stroke = if (fill == android.graphics.Color.BLACK) android.graphics.Color.WHITE else android.graphics.Color.BLACK
                    val cx = x * cell + (cell * 0.5f)
                    // Обводка
                    paint.style = Paint.Style.STROKE
                    paint.strokeWidth = strokeW
                    paint.color = stroke
                    cv.drawText(ch.toString(), cx, cy, paint)
                    // Заливка
                    paint.style = Paint.Style.FILL
                    paint.color = fill
                    cv.drawText(ch.toString(), cx, cy, paint)
                }
            }
        }
    }
    // --- Легенда (компактная, в несколько колонок)
    if (wantLegend && layoutItems.isNotEmpty()) {
        val baseY = outHimg.toFloat() + margin
        paint.textSize = max(16f, cell * 1.2f)
        paint.textAlign = Paint.Align.LEFT
        val ascent = -paint.fontMetrics.ascent
        layoutItems.forEach { item ->
            var x = item.x
            val y = baseY + item.y
            // swatch
            paint.style = Paint.Style.FILL
            paint.color = item.swatch
            cv.drawRect(x, y - swatch*0.75f, x + swatch, y - swatch*0.75f + swatch, paint)
            x += swatch + hGap
            // symbol (чёрный)
            paint.color = android.graphics.Color.BLACK
            cv.drawText(item.symbol.toString(), x, y + ascent*0.15f, paint)
            x += item.widths[0] + hGap
            // code
            cv.drawText(item.code, x, y + ascent*0.15f, paint); x += item.widths[1] + hGap
            // name (обрезка на всякий случай)
            cv.drawText(item.name, x, y + ascent*0.15f, paint); x += item.widths[2] + hGap
            // percent
            cv.drawText(item.percent, x, y + ascent*0.15f, paint)
        }
    }
    return out
    }
// Вспомогательная структура для укладки элементов легенды
private data class LegendItem(
    val x: Float,
    val y: Float,
    val swatch: Int,
    val symbol: Char,
    val code: String,
    val name: String,
    val percent: String,
    val widths: FloatArray
    )

