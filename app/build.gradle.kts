plugins {
    id("com.android.application")
    kotlin("android")
    kotlin("plugin.serialization")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.appforcross.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.appforcross.editor"
        minSdk = 31
        targetSdk = 35

        versionCode = 1
        versionName = "1.0.0"

        vectorDrawables.useSupportLibrary = true
    }

    buildTypes {
        debug {
            // В отладке ресурсы/код не сжимаем
            isMinifyEnabled = false
            isShrinkResources = false
        }
        release {
            // В релизе включаем R8 и ресурс-ши́нкер
            isMinifyEnabled = true
            isShrinkResources = true

            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )

            // Если подпись через подп.конфиг — раскомментируй и настрой:
            // signingConfig = signingConfigs.getByName("release")
        }
    }

    // Если нужны подписи из keystore:
    // signingConfigs {
    //     create("release") {
    //         storeFile = file("keystore.jks")
    //         storePassword = System.getenv("KEYSTORE_PASSWORD")
    //         keyAlias = System.getenv("KEY_ALIAS")
    //         keyPassword = System.getenv("KEY_PASSWORD")
    //     }
    // }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    // Версию компилятора Compose подбери под свою сборку (или вынеси в версио‑каталог)
    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.14"
    }

    // Java 17 (рекомендуется для AGP 8.x + Compose)
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
        freeCompilerArgs += listOf(
            "-Xjvm-default=all"
        )
    }

    packaging {
        resources {
            // Часто помогает от дубликатов лицензий/метаданных
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }

    lint {
        // В релиз перед выкладкой можно сделать true
        abortOnError = false
        checkReleaseBuilds = true
    }
}

dependencies {
    // Compose BOM — подтягивает согласованные версии UI/Material и т.п.
    implementation(platform("androidx.compose:compose-bom:2024.09.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    debugImplementation("androidx.compose.ui:ui-tooling")

    implementation("androidx.activity:activity-compose:1.9.2")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.6")

    // Документы/SAF
    implementation("androidx.documentfile:documentfile:1.0.1")

    // DataStore (если используешь SettingsRepository)
    implementation("androidx.datastore:datastore-preferences:1.1.1")

    // Kotlinx Serialization JSON
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")
    implementation("androidx.core:core-splashscreen:1.0.1")
    implementation("com.google.android.material:material:1.12.0")

    // (Опционально) Coil для картинок
    // implementation("io.coil-kt:coil-compose:2.6.0")

    // Тесты
    testImplementation(kotlin("test"))
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
    androidTestImplementation(platform("androidx.compose:compose-bom:2024.09.01"))
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")

    implementation("androidx.compose.material:material-icons-extended")

    implementation(project(":core"))
}
