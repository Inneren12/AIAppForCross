
package com.appforcross.i18n

import androidx.compose.runtime.staticCompositionLocalOf

val LocalStrings = staticCompositionLocalOf<Strings> {
    error("Strings not provided")
    }
