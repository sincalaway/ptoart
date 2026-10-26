package com.sincalaway.ptoart

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.SeekBar
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.lifecycleScope
import com.sincalaway.ptoart.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.max

class MainActivity : AppCompatActivity() {
    private lateinit var ui: ActivityMainBinding
    private var photo: Bitmap? = null
    private var result: Bitmap? = null
    private var cartoonizer: Pair<ModelSource, Cartoonizer>? = null
    private var styleModel: Pair<ModelSource, StyleModel>? = null
    // Imported files (optional overrides of the bundled models of the same name).
    private val modelFile by lazy { File(filesDir, "ptoart_weights.onnx") }
    private val upscalerFile by lazy { File(filesDir, "upscaler.onnx") }
    private var upscaler: Pair<ModelSource, Upscaler>? = null
    private val styleDir by lazy { File(filesDir, "styles").apply { mkdirs() } }

    /**
     * Models shipped in the APK under assets/models (release builds): keys "ptoart_weights",
     * "styles/<stem>" and "upscale/<stem>". Empty in development builds without models.
     */
    private val bundled: Map<String, ModelSource> by lazy {
        buildMap<String, ModelSource> {
            for (dir in listOf("", "styles", "upscale")) {
                val folder = if (dir.isEmpty()) "models" else "models/$dir"
                val names = runCatching { assets.list(folder) }.getOrNull() ?: emptyArray()
                for (name in names) {
                    if (!name.endsWith(".onnx")) continue
                    val key = (if (dir.isEmpty()) "" else "$dir/") + name.removeSuffix(".onnx")
                    put(key, ModelSource.Bundled(assets, "$folder/$name"))
                }
            }
        }
    }

    /** What the user can produce: face modes need the avatar model, styles need their own file. */
    private sealed class Mode(val label: String) {
        class Face(label: String, val model: ModelSource) : Mode(label)
        class Photo(label: String, val model: ModelSource) : Mode(label)
        class Style(label: String, val model: ModelSource) : Mode(label)
        override fun toString() = label
    }
    private var modes: List<Mode> = emptyList()
    private val currentMode get() = modes.getOrNull(ui.mode.selectedItemPosition)
    private var upscaleChoices: List<Pair<String, ModelSource>> = emptyList()

    private val styleNames = mapOf(
        "hayao" to "宫崎骏", "shinkai" to "新海诚", "paprika" to "今敏 Paprika",
        "celeba_distill" to "CelebA 蒸馏", "face_paint_512_v1" to "Face Paint v1",
        "face_paint_512_v2" to "Face Paint v2", "hayao_v3" to "宫崎骏 v3", "shinkai_v3" to "新海诚 v3",
        "ghibli_face" to "吉卜力人像", "jp_face" to "日漫二次元", "portrait_sketch" to "黑白素描",
        "dct_anime_bg" to "DCT 日漫", "dct_3d" to "DCT 3D", "dct_handdrawn_bg" to "DCT 手绘",
    )
    /** DCT-Net head models paint faces only; their *_bg twins are the whole-image styles. */
    private val headOnly = setOf("dct_anime", "dct_handdrawn")

    private val pickModel = registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isEmpty()) return@registerForActivityResult
        lifecycleScope.launch {
            busy(true, "正在导入模型…")
            val messages = withContext(Dispatchers.IO) {
                uris.map { uri ->
                    val name = displayName(uri) ?: "model.onnx"
                    runCatching {
                        val tmp = File(filesDir, "model.tmp")
                        contentResolver.openInputStream(uri)!!.use { i -> tmp.outputStream().use { i.copyTo(it) } }
                        when (ModelInspector.kindOf(tmp)) { // validates before replacing anything
                            ModelKind.AVATAR -> {
                                cartoonizer?.second?.close(); cartoonizer = null
                                check(tmp.renameTo(modelFile)) { "无法保存模型" }
                                "已导入卡通头像模型"
                            }
                            ModelKind.UPSCALER -> {
                                cartoonizer?.second?.upscaler = null
                                upscaler?.second?.close(); upscaler = null
                                check(tmp.renameTo(upscalerFile)) { "无法保存模型" }
                                "已导入超分辨率模型"
                            }
                            ModelKind.STYLE -> {
                                val stem = name.substringBeforeLast('.').ifBlank { "style" }
                                val target = File(styleDir, "$stem.onnx")
                                // Replaces an earlier import or overrides the bundled style of the same name.
                                if (styleModel?.first?.stem == stem) { styleModel?.second?.close(); styleModel = null }
                                if (cartoonizer?.first?.stem == stem) { cartoonizer?.second?.close(); cartoonizer = null }
                                check(tmp.renameTo(target)) { "无法保存模型" }
                                "已导入风格：${styleNames[stem] ?: stem}"
                            }
                        }
                    }.getOrElse { "$name 无效：${it.message}" }
                }
            }
            busy(false)
            toast(messages.joinToString("\n"))
            refreshModelState()
        }
    }

    private val pickPhoto = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri ?: return@registerForActivityResult
        lifecycleScope.launch {
            val bmp = withContext(Dispatchers.IO) { runCatching { decode(uri) }.getOrNull() }
            if (bmp == null) { toast("无法读取这张图片"); return@launch }
            photo = bmp; result = null
            ui.image.setImageBitmap(bmp)
            ui.save.isEnabled = false
            refreshModelState()
        }
    }

    private val saveResult = registerForActivityResult(ActivityResultContracts.CreateDocument("image/png")) { uri ->
        val bmp = result ?: return@registerForActivityResult
        uri ?: return@registerForActivityResult
        lifecycleScope.launch(Dispatchers.IO) {
            val ok = runCatching {
                contentResolver.openOutputStream(uri)!!.use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
            }.isSuccess
            withContext(Dispatchers.Main) { toast(if (ok) "已保存" else "保存失败") }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ui = ActivityMainBinding.inflate(layoutInflater)
        setContentView(ui.root)
        ui.importModel.setOnClickListener { pickModel.launch(arrayOf("*/*")) }
        ui.mode.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) = refreshModelState(rebuild = false)
            override fun onNothingSelected(p: AdapterView<*>?) {}
        }
        ui.superRes.setOnCheckedChangeListener { _, _ -> refreshModelState(rebuild = false) }
        ui.pick.setOnClickListener { pickPhoto.launch("image/*") }
        ui.convert.setOnClickListener { convert() }
        ui.save.setOnClickListener { saveResult.launch("ptoart-${System.currentTimeMillis()}.png") }
        ui.strength.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar?, p: Int, fromUser: Boolean) {
                ui.strengthLabel.text = getString(R.string.strength_fmt, p)
            }
            override fun onStartTrackingTouch(s: SeekBar?) {}
            override fun onStopTrackingTouch(s: SeekBar?) {}
        })
        ui.strengthLabel.text = getString(R.string.strength_fmt, ui.strength.progress)
        refreshModelState()
    }

    private fun refreshModelState(rebuild: Boolean = true) {
        if (rebuild) {
            val previous = currentMode?.label
            modes = buildList {
                // AnimeGANv2 face painter keeps faces natural (better on children); listed first.
                val looks = listOf("face_paint_512_v2" to "动漫脸（推荐）", "dct_anime" to "DCT 日漫",
                    "dct_3d" to "DCT 3D", "dct_handdrawn" to "DCT 手绘", "jp_face" to "日漫二次元",
                    "ghibli_face" to "吉卜力人像", "portrait_sketch" to "黑白素描")
                val styles = styleModels()
                for ((stem, look) in looks) {
                    val painter = styles[stem] ?: continue
                    add(Mode.Face("卡通头像 · $look", painter))
                    add(Mode.Photo("整张照片 · $look", painter))
                }
                avatarModel()?.let {
                    add(Mode.Face("卡通头像 · 经典卡通", it))
                    add(Mode.Photo("整张照片 · 经典卡通", it))
                }
                for ((stem, model) in styles) {
                    if (stem in headOnly) continue
                    add(Mode.Style("整图风格：${styleNames[stem] ?: stem}", model))
                }
            }
            ui.mode.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, modes)
            modes.indexOfFirst { it.label == previous }.takeIf { it >= 0 }?.let { ui.mode.setSelection(it) }

            val previousUpscaler = upscaleChoices.getOrNull(ui.upscaleChoice.selectedItemPosition)?.first
            upscaleChoices = buildList<Pair<String, ModelSource>> {
                if (upscalerFile.isFile) add("超分：已导入的模型" to ModelSource.Imported(upscalerFile))
                bundled["upscale/anime"]?.let { add("超分：动漫（快）" to it) }
                bundled["upscale/general"]?.let { add("超分：通用（照片）" to it) }
            }
            ui.upscaleChoice.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item,
                upscaleChoices.map { it.first })
            upscaleChoices.indexOfFirst { it.first == previousUpscaler }.takeIf { it >= 0 }
                ?.let { ui.upscaleChoice.setSelection(it) }
        }
        val styles = modes.count { it is Mode.Style }
        ui.modelState.text = when {
            modes.isEmpty() -> getString(R.string.model_missing)
            else -> getString(R.string.model_ready_fmt,
                if (modes.any { it is Mode.Face }) "已就绪" else "未找到", styles,
                if (upscaleChoices.isNotEmpty()) "已就绪" else "未找到")
        }
        ui.mode.visibility = if (modes.isEmpty()) View.GONE else View.VISIBLE
        ui.transparent.visibility = if (currentMode is Mode.Face) View.VISIBLE else View.GONE
        val superResShown = upscaleChoices.isNotEmpty() && currentMode != null && currentMode !is Mode.Style
        ui.superRes.visibility = if (superResShown) View.VISIBLE else View.GONE
        ui.upscaleChoice.visibility =
            if (superResShown && ui.superRes.isChecked && upscaleChoices.size > 1) View.VISIBLE else View.GONE
        ui.convert.isEnabled = currentMode != null && photo != null
    }

    /** The avatar generator: an imported ptoart_weights.onnx wins over the bundled one. */
    private fun avatarModel(): ModelSource? =
        if (modelFile.isFile) ModelSource.Imported(modelFile) else bundled["ptoart_weights"]

    /** Bundled styles plus imported ones, sorted by name; an import replaces the bundled file of that name. */
    private fun styleModels(): Map<String, ModelSource> {
        val found = sortedMapOf<String, ModelSource>()
        for ((key, model) in bundled) {
            if (key.startsWith("styles/")) found[key.removePrefix("styles/")] = model
        }
        styleDir.listFiles { f -> f.extension == "onnx" }?.forEach {
            found[it.nameWithoutExtension] = ModelSource.Imported(it)
        }
        return found
    }

    private fun convert() {
        val src = photo ?: return
        val mode = currentMode ?: return
        val strength = ui.strength.progress / 100f
        val transparent = ui.transparent.isChecked
        val upscaleModel = if (ui.superRes.isChecked && ui.superRes.visibility == View.VISIBLE) {
            upscaleChoices.getOrNull(ui.upscaleChoice.selectedItemPosition)?.second ?: upscaleChoices.firstOrNull()?.second
        } else null
        lifecycleScope.launch {
            busy(true, "正在转换…")
            val out = withContext(Dispatchers.Default) {
                runCatching {
                    val faceSource = when (mode) { is Mode.Face -> mode.model; is Mode.Photo -> mode.model; else -> null }
                    if (faceSource != null) {
                        val sr = upscaleModel?.let { upscalerFor(it) }
                        faceModel(faceSource).upscaler = sr
                    }
                    when (mode) {
                        is Mode.Face -> faceModel(mode.model).convertFace(src, strength, if (transparent) null else Color.WHITE)
                        is Mode.Photo -> faceModel(mode.model).convertPhoto(src, strength)
                        is Mode.Style -> {
                            val current = styleModel
                            val model = if (current?.first == mode.model) current.second else {
                                current?.second?.close()
                                StyleModel(mode.model).also { styleModel = mode.model to it }
                            }
                            model.stylize(src, strength)
                        }
                    }
                }
            }
            busy(false)
            out.onSuccess {
                result = it
                ui.image.setImageBitmap(it)
                ui.save.isEnabled = true
            }.onFailure { toast(it.message ?: "转换失败") }
        }
    }

    private fun faceModel(model: ModelSource): Cartoonizer {
        cartoonizer?.let { if (it.first == model) return it.second else it.second.close() }
        return Cartoonizer(model, painter = model != avatarModel()).also { cartoonizer = model to it }
    }

    private fun upscalerFor(model: ModelSource): Upscaler {
        upscaler?.let {
            if (it.first == model) return it.second
            cartoonizer?.second?.upscaler = null
            it.second.close()
        }
        return Upscaler(model).also { upscaler = model to it }
    }

    private fun displayName(uri: Uri): String? =
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use {
            if (it.moveToFirst()) it.getString(0) else null
        }

    private fun decode(uri: Uri): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it, null, bounds) }
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= 1600) sample *= 2
        val opts = BitmapFactory.Options().apply {
            inSampleSize = sample; inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val bmp = contentResolver.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it, null, opts) }!!
        val orientation = contentResolver.openInputStream(uri)!!.use {
            ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        }
        val m = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
            else -> return bmp
        }
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
    }

    private fun busy(on: Boolean, msg: String = "") {
        ui.progress.visibility = if (on) View.VISIBLE else View.GONE
        ui.status.text = msg
        listOf(ui.importModel, ui.pick, ui.convert, ui.mode, ui.upscaleChoice).forEach { it.isEnabled = !on }
        if (!on) refreshModelState(rebuild = false)
    }

    private fun toast(s: String) = Toast.makeText(this, s, Toast.LENGTH_LONG).show()

    override fun onDestroy() {
        cartoonizer?.second?.close()
        styleModel?.second?.close()
        upscaler?.second?.close()
        super.onDestroy()
    }
}
