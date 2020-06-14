package com.hotmail.birdpredictor.view;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.provider.MediaStore;
import android.view.Surface;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import com.hotmail.birdpredictor.R;
import com.hotmail.birdpredictor.model.Classifier;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

/**
 * Reference code: https://www.youtube.com/watch?v=s1aOlr3vbbk
 */
public class MainActivity extends AppCompatActivity {
    private static final int CAMERA_PERM_CODE = 87;
    private static final int CAMERA_REQUEST_CODE = 19;
    private static final int LIST_REQUEST_CODE = 100;
    private Integer sensorOrientation;
    private Classifier classifier;
    private Classifier.Recognition recognition;
    private Handler handler;
    private HandlerThread handlerThread;
    private String currentImgPath;
    private AssetManager assetManager;
    private Button cameraBtn;
    private Button predictBtn;
    private Button testBtn;
    private ImageView imageView;
    private Bitmap capturedImage;
    private TextView predictBirdTextView;
    private Intent toList;
    private String assetTestFolderpath= "TestImg/";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        assetManager = getAssets();
        predictBirdTextView = (TextView) findViewById(R.id.predictBirdTextView);
        imageView = (ImageView) findViewById(R.id.imageView);
        cameraBtn = (Button) findViewById(R.id.cameraBtn);
        testBtn = (Button) findViewById(R.id.testBtn);
        testBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                goToListView();
            }
        });
        predictBtn = (Button) findViewById(R.id.predictBtn);
        predictBtn.setEnabled(false);
        cameraBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                    askCameraPermissions();
            }
        });
        int numOFthreads = 10;
        try {
            classifier= Classifier.create(this, Classifier.Model.MY_MOBILE_NET_V2,Classifier.Device.CPU,numOFthreads);
        } catch (IOException e) {
            e.printStackTrace();
        }
        predictBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                predictResult();
            }
        });
    }
    private void predictResult(){
        predictBirdTextView.setText("");
        int rotation =90;
        sensorOrientation = rotation - getScreenOrientation();
        List<Classifier.Recognition> results =classifier.recognizeImage(capturedImage, sensorOrientation);
        String fullResult="";
        if (results != null && results.size() >= 3) {
            for (int i=0; i<3;i++){
                recognition = results.get(i);
                if ((recognition != null) && (recognition.getConfidence() != null)) {
                    if (recognition.getTitle() != null) {
                        String concat=recognition.getId()+" "+"("+String.format("%.2f", (100 * recognition.getConfidence())) + "%"+")"+"\n \n";
                        fullResult=fullResult+concat;
                    }
                }
            }
        }
        predictBirdTextView.setText(fullResult);
    }
    private void goToListView(){
        toList = new Intent(this,ListActivity2.class);
        startActivityForResult(toList,LIST_REQUEST_CODE);
    }
    private void processListIntent(Intent data){
        String pictureName = data.getStringExtra(ListActivity2.LIST_INTENT_KEY);
        predictBirdTextView.setText("");
        InputStream file = null;
        try {
            file = assetManager.open(assetTestFolderpath+pictureName);
        } catch (IOException e) {
            e.printStackTrace();
        }
        Bitmap bitmap = BitmapFactory.decodeStream(file);
        Bitmap resizedImg = Bitmap.createScaledBitmap(bitmap, 224, 224, false);
        imageView.setImageBitmap(resizedImg);
        capturedImage=resizedImg;
        predictBtn.setEnabled(true);
    }

    private void processTakePictureIntent(){
        //Image read from camera is not so good in prediction
        predictBirdTextView.setText("");
        File f = new File(currentImgPath);
        imageView.setImageURI(Uri.fromFile(f));
        Bitmap image = BitmapFactory.decodeFile(f.getAbsolutePath());
        Bitmap resizedImg = Bitmap.createScaledBitmap(image, 224, 224, false);
        capturedImage = resizedImg.copy(Bitmap.Config.ARGB_8888, true);
        f.delete();
        predictBtn.setEnabled(true);
    }
    private void askCameraPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA,Manifest.permission.WRITE_EXTERNAL_STORAGE}, CAMERA_PERM_CODE);
        } else {
            dispatchTakePictureIntent();
        }
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == CAMERA_PERM_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                dispatchTakePictureIntent();
            } else {
                Toast.makeText(this, "Camera Permission is Required to Use camera.", Toast.LENGTH_SHORT).show();
            }
        }
    }
    private void dispatchTakePictureIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        // Ensure that there's a camera activity to handle the intent
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            // Create the File where the photo should go
            File photoFile = null;
            try {
                photoFile = createImageFile();
            } catch (IOException ex) {
                // Error occurred while creating the File
                ex.printStackTrace();
            }
            // Continue only if the File was successfully created
            if (photoFile != null) {
                Uri photoURI = FileProvider.getUriForFile(this,
                        "com.example.android.fileprovider",
                        photoFile);
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                startActivityForResult(takePictureIntent, CAMERA_REQUEST_CODE);
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == CAMERA_REQUEST_CODE) {
            if (resultCode == Activity.RESULT_OK) {
                processTakePictureIntent();
            }
        }
        if (requestCode == LIST_REQUEST_CODE){
            if(resultCode==RESULT_OK){
                processListIntent(data);
            }
        }
    }

    private File createImageFile() throws IOException{
        String  timeStamp=new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageName="IMG_"+timeStamp+"_";
        File storageDir=getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File imageFile =File.createTempFile(imageName,".jpg",storageDir);
        currentImgPath=imageFile.getAbsolutePath();
        return imageFile;
    }

    protected int getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }
    @Override
    public synchronized void onStart() {
        //LOGGER.d("onStart " + this);
        super.onStart();
    }

    @Override
    public synchronized void onResume() {
        //LOGGER.d("onResume " + this);
        super.onResume();

        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
    }

    @Override
    public synchronized void onPause() {
       // LOGGER.d("onPause " + this);

        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (final InterruptedException e) {
            //LOGGER.e(e, "Exception!");
        }

        super.onPause();
    }

    @Override
    public synchronized void onStop() {
        //LOGGER.d("onStop " + this);
        super.onStop();
    }

}


