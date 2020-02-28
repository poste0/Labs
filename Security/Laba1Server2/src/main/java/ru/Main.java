package ru;

import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {
        while(true) {
            SocketProxy p = new SocketProxyImpl(args);
            p.accept();
            p.cleanup();
        }
    }
}
