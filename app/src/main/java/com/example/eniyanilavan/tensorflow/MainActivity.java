package com.example.eniyanilavan.tensorflow;

import android.annotation.SuppressLint;
import android.os.Looper;
import android.support.v7.app.AppCompatActivity;


import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Size;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.List;

import static org.opencv.imgproc.Imgproc.INTER_CUBIC;
import static org.opencv.imgproc.Imgproc.resize;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{
    static {
        System.loadLibrary("tensorflow_inference");
    }
    JavaCameraView jj;
    Mat mRgba;
    TextView t;

    BaseLoaderCallback bmLoader=new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch(status) {
                case BaseLoaderCallback.SUCCESS:
                    jj.enableView();
                    break;
                default : super.onManagerConnected(status);

            }


        }
    };
    private TensorFlowInferenceInterface tensorFlowInferenceInterface;
    private static final String MODEL_NAME = "file:///android_asset/frozen_tflearn.pb";
    private static final String INPUT_NAME = "input/X";
    private static final String OUTPUT_NAME = "output/Softmax";
    private static final int[] INPUT_SIZE = {1,32,32,1};


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        tensorFlowInferenceInterface = new TensorFlowInferenceInterface(getAssets(),MODEL_NAME);
        jj=(JavaCameraView)findViewById(R.id.javaCamera);
        jj.setVisibility(SurfaceView.VISIBLE);
        jj.setCvCameraViewListener(this);
        t = (TextView)findViewById(R.id.textView);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if(jj!=null)
        {
            jj.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()){
            Log.d("MA","loaded");
            bmLoader.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        else {
            Log.d("MA","not loaded");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9,this,bmLoader);
        }

    }
    @Override
    protected void onPause() {

        super.onPause();
        if(jj!=null)
        {
            jj.disableView();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba= new Mat(width,height, CvType.CV_8UC1);

    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();

    }

    @Override
    public Mat onCameraFrame(Mat inputFrame) {
        return null;
    }

    @SuppressLint("SetTextI18n")
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.gray();
        Mat a1 = new Mat(32,32,1);
        resize(mRgba,a1,a1.size(),0,0,INTER_CUBIC);
        float a[] = new float[a1.rows()*a1.cols()];
        int k = 0;
        for(int i=1;i<a1.rows();i++) {
            for (int j = 1; j < a1.cols(); j++)
            {
                a[k] = (float) a1.get(i,j)[0];
                k++;
            }
        }
        float[] res = {0,0};
        tensorFlowInferenceInterface = new TensorFlowInferenceInterface(getAssets(),MODEL_NAME);
        tensorFlowInferenceInterface.feed(INPUT_NAME,a,1,32,32,1);
        tensorFlowInferenceInterface.run(new String[] {OUTPUT_NAME});
        tensorFlowInferenceInterface.fetch(OUTPUT_NAME, res);
        if (res[0]>res[1])
        {
            Log.d("predicted","cat");
        }
        else
        {
            Log.d("predicted","dog");
        }
        return mRgba;
    }
}
