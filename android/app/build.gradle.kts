plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val ksPath: String? = System.getenv("PTOART_KEYSTORE")

android {
    namespace = "com.sincalaway.ptoart"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.sincalaway.ptoart"
        minSdk = 24
        targetSdk = 34
        versionCode = (System.getenv("PTOART_VERSION_CODE") ?: "12").toInt()
        versionName = System.getenv("PTOART_VERSION_NAME") ?: "0.3.10"
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64") }
    }

    signingConfigs {
        if (ksPath != null) {
            create("release") {
                storeFile = file(ksPath)
                storePassword = System.getenv("PTOART_KEYSTORE_PASSWORD")
                keyAlias = System.getenv("PTOART_KEY_ALIAS")
                keyPassword = System.getenv("PTOART_KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // Without a release keystore (CI secrets), sign with the debug key so the
            // APK is still installable by sideloading.
            signingConfig = if (ksPath != null) signingConfigs.getByName("release")
                            else signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { viewBinding = true }
    lint {
        abortOnError = false
        checkReleaseBuilds = false
    }
    packaging { jniLibs { useLegacyPackaging = false } }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.2")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
    implementation("androidx.exifinterface:exifinterface:1.3.7")
    implementation("com.google.android.material:material:1.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-play-services:1.8.1")
    // On-device inference: cartoon generator (user-supplied ONNX), face + person mask.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    implementation("com.google.mlkit:face-detection:16.1.7")
    implementation("com.google.mlkit:segmentation-selfie:16.0.0-beta6")
}
