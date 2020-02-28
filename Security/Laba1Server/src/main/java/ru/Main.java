package ru;

import javax.imageio.ImageIO;
import javax.net.ServerSocketFactory;
import javax.net.ssl.HttpsURLConnection;
import javax.net.ssl.SSLServerSocket;
import javax.net.ssl.SSLServerSocketFactory;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws IOException {
        //System.setProperty("javax.net.ssl.keyStore", "C:/Users/sergei/IdeaProjects/LabsSecurity/examplestore");
        //System.setProperty("javax.net.ssl.keyStorePassword", "javajava");
        System.out.println("The server is working");
        while(true) {
            System.out.println("Waiting");
            SocketProxy p = null;
            try {
                p = new SocketProxyImpl(args);
                p.accept();
                p.createNoise();
                p.sendNext();
            } catch (Exception e) {
                System.out.println("An error happened \n" + e.getMessage());
                e.printStackTrace();
            }
            finally {
                p.cleanup();
            }
        }
    }
}
