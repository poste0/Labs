package ru;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        final int m;
        List<Integer> nodes = new ArrayList<Integer>();

        m = scanner.nextInt();
        int node;
        for (int i = 0; i < m; i++) {
            do {
                node = scanner.nextInt();
            }
            while (node > Math.pow(2, m));

            nodes.add(node);
        }


    }
}
