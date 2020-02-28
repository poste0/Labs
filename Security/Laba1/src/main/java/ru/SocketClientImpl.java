package ru;

import javax.imageio.ImageIO;
import javax.net.SocketFactory;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.net.Socket;

public class SocketClientImpl extends AbstractSocketClientImpl {

    public SocketClientImpl(String[] args){
        super(args);
        try {
            socketFactory = SocketFactory.getDefault();
            socket = socketFactory.createSocket(proxy, Integer.parseInt(proxyPort));
        }
        catch (IllegalArgumentException | IOException e){
            e.printStackTrace();
        }
    }
}
