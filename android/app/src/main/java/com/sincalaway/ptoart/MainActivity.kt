package com.sincalaway.ptoart

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.view.View
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
    private var cartoonizer: Cartoonizer? = null
    private val modelFile by lazy { File(filesDir, "photo2cartoon_weights.onnx") }

    private val pickModel = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@registerForActivityResult
        lifecycleScope.launch {
            busy(true, "正在导入模型…")
            val ok = withContext(Dispatchers.IO) {
                runCatching {
                    val tmp = File(filesDir, "model.tmp")
                    contentResolver.openInputStream(uri)!!.use { i -> tmp.outputStream().use { i.copyTo(it) } }
                    cartoonizer?.close(); cartoonizer = null
                    Cartoonizer(tmp).close() // validate before replacing the current model
                    tmp.renameTo(modelFile)
                }
            }
            busy(false)
            ok.onFailure { toast("模型无效：${it.message}") }
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

    private fun refreshModelState() {
        val has = modelFile.isFile
        ui.modelState.text = getString(if (has) R.string.model_ready else R.string.model_missing)
        ui.convert.isEnabled = has && photo != null
    }

    private fun convert() {
        val src = photo ?: return
        lifecycleScope.launch {
            busy(true, "正在转换…")
            val out = withContext(Dispatchers.Default) {
                runCatching {
                    val c = cartoonizer ?: Cartoonizer(modelFile).also { cartoonizer = it }
                    val bg = if (ui.transparent.isChecked) null else Color.WHITE
                    c.convert(src, ui.strength.progress / 100f, bg)
                }
            }
            busy(false)
            out.onSuccess {
                result = it
                ui.image.setImageBitmap(Bitmap.createScaledBitmap(it, 768, 768, true))
                ui.save.isEnabled = true
            }.onFailure { toast(it.message ?: "转换失败") }
        }
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
        listOf(ui.importModel, ui.pick, ui.convert).forEach { it.isEnabled = !on }
        if (!on) refreshModelState()
    }

    private fun toast(s: String) = Toast.makeText(this, s, Toast.LENGTH_LONG).show()

    override fun onDestroy() {
        cartoonizer?.close()
        super.onDestroy()
    }
}
