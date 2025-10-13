// C
package com.appforcross.editor.ui.tabs

import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material3.*
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.appforcross.editor.EditorViewModel
import com.appforcross.editor.model.*
import com.appforcross.editor.ui.components.ApplyBar
import com.appforcross.editor.ui.components.Section
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.TextButton
import androidx.compose.material3.AssistChip
import androidx.compose.material3.OutlinedButton
import com.appforcross.i18n.LocalStrings

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun PaletteTab(vm: EditorViewModel) {
    val st by vm.state.collectAsState()
    val p = st.palette
    val scroll = rememberScrollState()
    val S = LocalStrings.current
    Column(Modifier.fillMaxSize().verticalScroll(scroll)) {
        // --- Выбор активной палитры из assets (реактивно) ---
        val palettes = remember { vm.getPalettes() }
        var expanded by rememberSaveable { mutableStateOf(false) }
        val activeId by vm.activePaletteId.collectAsState()
        val activeName = remember(palettes, activeId) {
            palettes.firstOrNull { it.id == activeId }?.name ?: activeId
        }

        Section(S.palette.assetsTitle) {
            Box {
                OutlinedButton(onClick = { expanded = true }) {
                    Text(activeName, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    Spacer(Modifier.width(8.dp))
                    Icon(if (expanded) Icons.Filled.ExpandLess else Icons.Filled.ExpandMore, contentDescription = null)
                }
                DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                    palettes.forEach { pm ->
                        DropdownMenuItem(
                            text = { Text(pm.name) },
                            onClick = {
                                vm.setActivePalette(pm.id)
                                expanded = false
                            }
                        )
                    }
                }
            }
        }

        Section(S.palette.maxColors) {
            OutlinedTextField(
                value = p.maxColors.toString(),
                onValueChange = { v -> v.toIntOrNull()?.let { k -> vm.updatePalette { st -> st.copy(maxColors = k.coerceAtLeast(0)) } } },
                label = { Text(S.palette.zeroMeansNoLimit) },
                modifier = Modifier.fillMaxWidth()
            )
        }

        Section(S.palette.metricTitle) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(selected = p.metric == ColorMetric.DE2000, onClick = { vm.updatePalette { it.copy(metric = ColorMetric.DE2000) } }, label = { Text(S.palette.metricDE2000) })
                FilterChip(selected = p.metric == ColorMetric.DE76, onClick = { vm.updatePalette { it.copy(metric = ColorMetric.DE76) } }, label = { Text(S.palette.metricDE76) })
                FilterChip(selected = p.metric == ColorMetric.OKLAB, onClick = { vm.updatePalette { it.copy(metric = ColorMetric.OKLAB) } }, label = { Text(S.palette.metricOKLAB) })
            }
        }

        Section(S.palette.ditheringTitle) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(selected = p.dithering == DitheringType.NONE, onClick = { vm.updatePalette { it.copy(dithering = DitheringType.NONE) } }, label = { Text(S.palette.ditheringNone) })
                FilterChip(selected = p.dithering == DitheringType.FLOYD_STEINBERG, onClick = { vm.updatePalette { it.copy(dithering = DitheringType.FLOYD_STEINBERG) } }, label = { Text(S.palette.ditheringFs) })
                FilterChip(selected = p.dithering == DitheringType.ATKINSON, onClick = { vm.updatePalette { it.copy(dithering = DitheringType.ATKINSON) } }, label = { Text(S.palette.ditheringAtkinson) })
            }
            Spacer(Modifier.height(4.dp))
            Text(S.palette.kmeansNote, style = MaterialTheme.typography.bodySmall)
        }

        // --- Секция «Нитки» после «Применить» ---
            val threads by vm.threads.collectAsState()
        LaunchedEffect(threads) {
            Log.d("PaletteTab", "UI threads[${threads.size}]: " +
                    threads.joinToString { "${it.code}:${it.percent}%(${it.count})" })
        }

            val symbolDraft by vm.symbolDraft.collectAsState()
            val symbolsApplied by vm.symbolsPreview.collectAsState()
            var showThreads by rememberSaveable { mutableStateOf(true) }
            var editCode by rememberSaveable { mutableStateOf<String?>(null) }
            var editChar by rememberSaveable { mutableStateOf("") }
            // Набор быстрых символов (для подбора)
            val quickSymbols = remember {
                listOf('●','○','■','□','▲','△','◆','◇','★','☆','✚','✖','✳','◼','◻','✦','✧',
                    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                    'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                    '0','1','2','3','4','5','6','7','8','9')
            }

        // Локальные параметры оценки длины (не пишем в EditorState)
        var fabricCountText by rememberSaveable { mutableStateOf("14") } // стежков/дюйм (Aida 14)
        var strandsText by rememberSaveable { mutableStateOf("2") }      // число сложений
        var wasteText by rememberSaveable { mutableStateOf("10") }        // % потерь
        val fabricCount = fabricCountText.toIntOrNull()?.coerceIn(6, 28) ?: 14
        val strands = strandsText.toIntOrNull()?.coerceIn(1, 6) ?: 2
        val waste = wasteText.toIntOrNull()?.coerceIn(0, 50) ?: 10
        Section(S.palette.threadsTitle) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(S.palette.estimateParams, style = MaterialTheme.typography.bodyMedium)
                Spacer(Modifier.weight(1f))
                IconButton(onClick = { showThreads = !showThreads }) {
                    Icon(
                        imageVector = if (showThreads) Icons.Filled.ExpandLess else Icons.Filled.ExpandMore,
                        contentDescription = null
                    )
                }
            }
            if (showThreads) {
                // Параметры расчёта длины
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                    OutlinedTextField(
                        value = fabricCountText,
                        onValueChange = { fabricCountText = it.filter { ch -> ch.isDigit() } },
                        label = { Text(S.palette.fabricStPerInch) },
                        modifier = Modifier.weight(1f)
                    )
                    OutlinedTextField(
                        value = strandsText,
                        onValueChange = { strandsText = it.filter { ch -> ch.isDigit() } },
                        label = { Text(S.palette.strands) },
                        modifier = Modifier.weight(1f)
                    )
                    OutlinedTextField(
                        value = wasteText,
                        onValueChange = { wasteText = it.filter { ch -> ch.isDigit() } },
                        label = { Text(S.palette.wastePct) },
                        modifier = Modifier.weight(1f)
                    )
                }
                Spacer(Modifier.height(8.dp))
                if (threads.isEmpty()) {
                    Text(S.palette.pressApplyToCalcThreads, style = MaterialTheme.typography.bodySmall)
                } else {
                    // Тумблер «Авто-символы» — влияет на поведение кнопки «Задать символы»
                    val auto by vm.autoSymbolsEnabled.collectAsState()
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Checkbox(checked = auto, onCheckedChange = { vm.setAutoSymbolsEnabled(it) })
                        Text(S.palette.autoSymbols)
                    }
                    Spacer(Modifier.height(4.dp))
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        threads.forEach { t ->
                            val ch = symbolDraft[t.code] ?: symbolsApplied[t.code]
                            val pitchInch = 1.0 / fabricCount.toDouble()
                            val pitchMm = 25.4 * pitchInch
                            val pathPerStitchMm = 2.8 * pitchMm
                            val lenMm = t.count * pathPerStitchMm * strands * (1.0 + waste / 100.0)
                            val lenM = lenMm / 1000.0
                            ListItem(
                                leadingContent = {
                                    Box(
                                        Modifier
                                            .size(24.dp)
                                            .clip(CircleShape)
                                            .background(Color(t.argb))
                                            .border(1.dp, MaterialTheme.colorScheme.outline, CircleShape)
                                    )
                                                 },
                                headlineContent = {
                                    Text("${t.code} · ${t.name}", maxLines = 1, overflow = TextOverflow.Ellipsis)
                                                  },
                                supportingContent = {
                                    Text("${t.percent}% · ~${"%.2f".format(lenM)} м")
                                                    },
                                trailingContent = {
                                    OutlinedButton(onClick = {
                                        editCode = t.code
                                        editChar = ch?.toString() ?: ""
                                    }) {
                                        Text(ch?.toString() ?: "—")
                                    }
                                }
                            )
                            HorizontalDivider()
                        }
                    }
                }
            }
        }

            // Кнопка «Задать символы» — только если у всех ниток есть символ
            val allHaveSymbols = remember(threads, symbolDraft, symbolsApplied) {
                if (threads.isEmpty()) false else threads.all { (symbolDraft[it.code] ?: symbolsApplied[it.code]) != null }
            }
            if (allHaveSymbols) {
                Spacer(Modifier.height(8.dp))
                Button(onClick = { vm.autoAssignSymbolsIfEnabled() }) { Text(S.palette.setSymbols) }
            }

            // Диалог редактирования символа
            if (editCode != null) {
                AlertDialog(
                    onDismissRequest = { editCode = null },
                    title = { Text(S.palette.symbolFor(editCode!!)) },
                    text = {
                        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedTextField(
                                value = editChar,
                                onValueChange = { s -> if (s.length <= 1) editChar = s },
                                label = { Text(S.palette.symbol1char) },
                                singleLine = true
                            )
                            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                val quickSymbols = listOf(
                                    '●','○','■','□','▲','△','◆','◇','★','☆','✚','✖','✳','◼','◻','✦','✧',
                                    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                                    'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                                    '0','1','2','3','4','5','6','7','8','9'
                                )
                                quickSymbols.forEach { q ->
                                    AssistChip(onClick = { editChar = q.toString() }, label = { Text(q.toString()) })
                                }
                            }
                        }
                           },
                    confirmButton = {
                        TextButton(onClick = {
                            val c = editChar.firstOrNull()
                            if (c != null) vm.setSymbol(editCode!!, c)
                            editCode = null
                        }) { Text(LocalStrings.current.common.ok) }
                                    },
                    dismissButton = { TextButton(onClick = { editCode = null }) { Text(LocalStrings.current.common.cancel) } }
                )
            }

        Spacer(Modifier.height(12.dp))
        // Undo/Redo + Apply
        val canUndo by vm.canUndo.collectAsState()
        val canRedo by vm.canRedo.collectAsState()
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            OutlinedButton(enabled = canUndo, onClick = { vm.undo() }) { Text(LocalStrings.current.common.undo) }
            OutlinedButton(enabled = canRedo, onClick = { vm.redo() }) { Text(LocalStrings.current.common.redo) }
            Spacer(Modifier.weight(1f))
            ApplyBar(enabled = st.sourceImage != null && !st.isBusy) { vm.applyPaletteKMeans() }
        }
    }
}