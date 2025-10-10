package com.appforcross.editor.ui

import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.foundation.layout.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.appforcross.editor.EditorViewModel
import com.appforcross.editor.ui.tabs.*
import com.appforcross.i18n.LocalStrings

enum class EditorTab { IMPORT, PREPROCESS, SIZE, PALETTE, OPTIONS, PREVIEW }

@Composable
fun EditorScreen(vm: EditorViewModel) {
    var current by remember { mutableStateOf(EditorTab.IMPORT) }
    val S = LocalStrings.current

    Column(Modifier.fillMaxSize()) {
        Text(
            text = S.nav.editorTitle,
            style = MaterialTheme.typography.headlineMedium,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
        )

        val tabs = listOf(
            EditorTab.IMPORT to S.nav.tabImport,
            EditorTab.PREPROCESS to S.nav.tabPreprocess,
            EditorTab.SIZE to S.nav.tabSize,
            EditorTab.PALETTE to S.nav.tabPalette,
            EditorTab.OPTIONS to S.nav.tabOptions,
            EditorTab.PREVIEW to S.nav.tabPreview
        )

        ScrollableTabRow(
            selectedTabIndex = tabs.indexOfFirst { it.first == current },
            edgePadding = 8.dp
        ) {
            tabs.forEach { (tab, title) ->
                Tab(
                    selected = current == tab,
                    onClick = { current = tab },
                    text = { Text(title) }
                )
            }
        }

        Divider()

        when (current) {
            EditorTab.IMPORT -> ImportTab(vm)
            EditorTab.PREPROCESS -> PreprocessTab(vm)
            EditorTab.SIZE -> SizeTab(vm)
            EditorTab.PALETTE -> PaletteTab(vm)
            EditorTab.OPTIONS -> OptionsTab(vm)
            EditorTab.PREVIEW -> PreviewTab(vm)
        }
    }
}