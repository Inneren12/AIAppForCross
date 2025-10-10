//Hash 7ef703ae7a8b6289e7c99078aafd0dcb
package com.appforcross.editor.ui.tabs

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.appforcross.editor.EditorViewModel
import com.appforcross.editor.model.DenoiseLevel
import com.appforcross.editor.ui.components.ApplyBar
import com.appforcross.editor.ui.components.Section
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.rememberScrollState
import com.appforcross.i18n.LocalStrings
import androidx.compose.ui.Alignment

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PreprocessTab(vm: EditorViewModel) {
    val st by vm.state.collectAsState()
    val S = LocalStrings.current

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {

        Section(S.preprocess.sectionBrightContrast) {
            Text(S.preprocess.brightnessPct(st.preprocess.brightnessPct))
            Slider(
                value = st.preprocess.brightnessPct.toFloat(),
                onValueChange = { v -> vm.updatePreprocess { it.copy(brightnessPct = v.toInt()) } },
                valueRange = -100f..100f
            )
            Spacer(Modifier.height(8.dp))
            Text(S.preprocess.contrastPct(st.preprocess.contrastPct))
            Slider(
                value = st.preprocess.contrastPct.toFloat(),
                onValueChange = { v -> vm.updatePreprocess { it.copy(contrastPct = v.toInt()) } },
                valueRange = -100f..100f
            )
        }

        Section(S.preprocess.sectionGamma) {
            Text(S.preprocess.gammaValue(st.preprocess.gamma))
            Slider(
                value = st.preprocess.gamma,
                onValueChange = { v -> vm.updatePreprocess { it.copy(gamma = v) } },
                valueRange = 0.1f..3.0f
            )

            Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
                Checkbox(
                    checked = st.preprocess.autoLevels,
                    onCheckedChange = { c -> vm.updatePreprocess { it.copy(autoLevels = c) } }
                )
                Text(S.preprocess.autoLevels)
            }
        }

        Section(S.preprocess.sectionDenoise) {
            var expanded by remember { mutableStateOf(false) }
            ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = !expanded }) {
                OutlinedTextField(
                    readOnly = true,
                    value = when (st.preprocess.denoise) {
                        DenoiseLevel.NONE -> S.preprocess.denoiseNone
                        DenoiseLevel.LOW -> S.preprocess.denoiseLow
                        DenoiseLevel.MEDIUM -> S.preprocess.denoiseMedium
                        DenoiseLevel.HIGH -> S.preprocess.denoiseHigh
                    },
                    onValueChange = {},
                    label = { Text(S.preprocess.sectionDenoise) },
                    modifier = Modifier.menuAnchor().fillMaxWidth()
                )
                ExposedDropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                    DropdownMenuItem(text = { Text(S.preprocess.denoiseNone) }, onClick = {
                        vm.updatePreprocess { it.copy(denoise = DenoiseLevel.NONE) }; expanded = false
                    })
                    DropdownMenuItem(text = { Text(S.preprocess.denoiseLow) }, onClick = {
                        vm.updatePreprocess { it.copy(denoise = DenoiseLevel.LOW) }; expanded = false
                    })
                    DropdownMenuItem(text = { Text(S.preprocess.denoiseMedium) }, onClick = {
                        vm.updatePreprocess { it.copy(denoise = DenoiseLevel.MEDIUM) }; expanded = false
                    })
                    DropdownMenuItem(text = { Text(S.preprocess.denoiseHigh) }, onClick = {
                        vm.updatePreprocess { it.copy(denoise = DenoiseLevel.HIGH) }; expanded = false
                    })
                }
            }
        }

        Section(S.preprocess.sectionTonal) {
            Text(S.preprocess.tonalCompression(st.preprocess.tonalCompression))
            Slider(
                value = st.preprocess.tonalCompression,
                onValueChange = { v -> vm.updatePreprocess { it.copy(tonalCompression = v) } },
                valueRange = 0f..1f
            )
            Spacer(Modifier.height(8.dp))
            Text(S.preprocess.noteAppliedBefore)
        }

        Spacer(Modifier.weight(1f))
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
            ApplyBar(enabled = st.sourceImage != null && !st.isBusy) { vm.applyPreprocess() }
        }
    }
}