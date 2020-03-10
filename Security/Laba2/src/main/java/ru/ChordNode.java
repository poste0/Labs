package ru;

import java.util.List;

public class ChordNode {
    private Integer id;

    private List<Integer> fingerStarts;

    private Interval interval;

    private ChordNode node;

    private ChordNode successor;

    private ChordNode predecessor;

    public ChordNode(int n, int m){
        this.id = n;
        for (int i = 1; i < m; i++) {
            fingerStarts.add((int) ((n + Math.pow(2, i - 1)) % Math.pow(2, m)));
        }

    }

    private class Interval{
        private int from;

        private int to;
    }
}
