package com.hotmail.birdpredictor.view;

import android.content.Intent;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.hotmail.birdpredictor.R;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

public class ListActivity2 extends AppCompatActivity {
    protected static int LIST_ITEM_SELECT_CODE=5;
    private ListView listView;
    private Intent intentFromMain;
    protected static String LIST_INTENT_KEY="PICTURE_NAME";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_list2);
        listView=(ListView)findViewById(R.id.listView);

        intentFromMain = getIntent();//get the to List intent from main activity
        final ArrayList<String> stringList=new ArrayList<>();
        readAssets(stringList);
        ArrayAdapter arrAdaptor = new ArrayAdapter(this,android.R.layout.simple_list_item_1,stringList);
        listView.setAdapter(arrAdaptor);
        listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                Toast.makeText(ListActivity2.this,"Selected item is "+stringList.get(position).toString(),Toast.LENGTH_LONG).show();
                String value=stringList.get(position).toString();
                intentFromMain.putExtra(LIST_INTENT_KEY, value);
                setResult(RESULT_OK, intentFromMain);
                finish();
            }
        });
    }

    private void readAssets(ArrayList<String>list){
        //Read from Asset folder. Image predicted is very good
        AssetManager assetManager = getAssets();
        InputStream file = null;
        try {
            //file = assetManager.open("Black_Baza_new_blackBaza_1.jpg");
            String[] files = assetManager.list("TestImg");
            for(int i=0;i<files.length;i++){
                if (files[i].endsWith(".jpg")){
                    list.add(files[i]);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}