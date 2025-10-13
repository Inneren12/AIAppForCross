// Hash 126e620e9dd80c20ef981d1db6483994
package com.appforcross.editor.ui.tabs

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.appforcross.editor.EditorViewModel
import com.appforcross.editor.ui.components.ApplyBar
import com.appforcross.editor.ui.components.Section
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.ImageBitmap
import com.appforcross.core.io.ImageDecoder
import com.appforcross.core.image.DecodedImage
import android.content.Intent
import androidx.documentfile.provider.DocumentFile
import com.appforcross.i18n.LocalStrings
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.launch
import androidx.compose.runtime.rememberCoroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.compose.runtime.rememberCoroutineScope

@Composable
fun ImportTab(vm: EditorViewModel) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val S = LocalStrings.current
    val exports by vm.exports.collectAsState()
    var thumb: ImageBitmap? by remember { mutableStateOf(null) }
    var meta: Pair<Int, Int>? by remember { mutableStateOf(null) }
    // Локальная реализация декодера ядра (без синглтонов/хелперов, только внутри вкладки)
        val decoder = remember {
            object : ImageDecoder {
                override fun decode(bytes: ByteArray): DecodedImage {
                    val opts = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
                    val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)
                    val w = bmp.width
                    val h = bmp.height
                    val argb = IntArray(w * h)
                    bmp.getPixels(argb, 0, w, 0, 0, w, h)
                    return DecodedImage(w, h, argb)
                }
            }
        }

    val openDoc = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            scope.launch {
                // I/O: читаем байты вне UI
                val bytes = withContext(Dispatchers.IO) {
                    context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
                }
                if (bytes != null) {
                    // CPU: декод ядром (ваш локальный decoder) в фоне
                    val dec = withContext(Dispatchers.Default) { decoder.decode(bytes) }
                    val bmp = withContext(Dispatchers.Default) {
                        Bitmap.createBitmap(dec.width, dec.height, Bitmap.Config.ARGB_8888).apply {
                            setPixels(dec.argb, 0, dec.width, 0, 0, dec.width, dec.height)
                        }
                    }
                    val img = bmp.asImageBitmap()
                    val aspect = dec.width.toFloat() / dec.height.coerceAtLeast(1)
                    // На главном: фиксируем источник и подсказку сцены
                    withContext(Dispatchers.Main) {
                        vm.setSource(img, aspect) // ранний hint
                        thumb = img
                        meta = dec.width to dec.height
                    }
                }
            }
        }
    }

    Column(Modifier.fillMaxSize()) {
        Section(S.import.sectionTitle) {
            Text(S.import.description)
                Spacer(Modifier.height(12.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(
                        onClick = { openDoc.launch(arrayOf("image/*")) }
                    ) { Text(S.import.selectImage) }
                }
            Spacer(Modifier.height(8.dp))
            // ── Тумблер «Авто распознавание при импорте»
            // ── Тумблеры: авто‑распознавание и авто‑обработка (после импорта)
            val autoDetect by vm.autoDetectOnImport.collectAsState()
            val autoProcess by vm.autoProcessOnImport.collectAsState()
            Column {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text("Авто распознавание при импорте")
                    Switch(checked = autoDetect, onCheckedChange = vm::setAutoDetectOnImport)
                }
                Spacer(Modifier.height(6.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text("Авто обработка после импорта")
                    Switch(checked = autoProcess, onCheckedChange = vm::setAutoProcessOnImport)
                }
            }
                Spacer(Modifier.height(12.dp))
                if (thumb != null && meta != null) {
                    val (w, h) = meta!!
                    Text(S.import.sizePx(w, h))
                    Spacer(Modifier.height(8.dp))
                    // «Уменьшенная копия» для отображения: просто ограничиваем размеры виджета
                    Image(
                        bitmap = thumb!!,
                        contentDescription = S.import.miniPreview,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(160.dp),
                        contentScale = ContentScale.Fit
                    )

                    // ── Сводка AutoS: ширина в стежках и число цветов
                    val sum by vm.autoSummary.collectAsState()
                    if (sum != null) {
                        Spacer(Modifier.height(6.dp))
                        AssistChip(
                            onClick = { /* no-op */ },
                            label = { Text("Ширина: ${sum!!.widthSt} стежков · Цветов: ${sum!!.colors}") }
                        )
                    }

                    // ── Подсказка типа/уверенности от SmartSceneDetector (до обработки)
                    val hint by vm.importSceneHint.collectAsState()
                    if (hint != null) {
                        Spacer(Modifier.height(8.dp))
                        Card {
                            Column(Modifier.padding(12.dp)) {
                                Text("Тип: ${hint!!.kind}")
                                Text("Уверенность: ${"%.2f".format(hint!!.confidence)}")
                                val top = hint!!.top3.joinToString { (k, p) -> "$k=${"%.2f".format(p)}" }
                                Text("Top‑3: $top")
                                if (hint!!.widthStitches != null || hint!!.colorsSelected != null) {
                                    Spacer(Modifier.height(4.dp))
                                    Text("Ширина: ${hint!!.widthStitches ?: "—"} стежков")
                                    Text("Цветов (по палитре): ${hint!!.colorsSelected ?: "—"}")
                                }
                            }
                        }
                    }

                } else {
                    Text(S.import.imageNotSelected)
                }
            }

            // --- Последние экспорты (управление файлами) ---
            if (exports.isNotEmpty()) {
                Spacer(Modifier.height(16.dp))
                Section(S.import.recentExports) {
                    exports.take(5).forEach { uri ->
                        val name = remember(uri) { DocumentFile.fromSingleUri(context, uri)?.name ?: uri.lastPathSegment ?: "export" }
                        Row(
                            Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = androidx.compose.ui.Alignment.CenterVertically
                        ) {
                            Text(name, modifier = Modifier.weight(1f))
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                TextButton(onClick = {
                                    val mime = context.contentResolver.getType(uri) ?: "*/*"
                                    val intent = Intent(Intent.ACTION_VIEW)
                                        .setDataAndType(uri, mime)
                                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                                    runCatching { context.startActivity(intent) }
                                }) { Text(S.common.open) }
                                TextButton(onClick = {
                                    val mime = context.contentResolver.getType(uri) ?: "*/*"
                                    val intent = Intent(Intent.ACTION_SEND)
                                        .setType(mime)
                                        .putExtra(Intent.EXTRA_STREAM, uri)
                                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                                    runCatching { context.startActivity(Intent.createChooser(intent, S.common.share)) }
                                }) { Text(S.common.share) }
                                TextButton(onClick = { vm.deleteExport(context, uri) }) { Text(S.common.delete) }
                            }
                        }
                    }
                }
            }

            Spacer(Modifier.weight(1f))
            ApplyBar(enabled = false, onApply = { })
    }
}