package ru;

import java.util.List;

public class FingerTable {
    private Integer start;

    private Interval interval;

    private Integer node;


    public FingerTable(int n, int m, int i, List<Integer> nodes){
        this.start = calculateStart(n, m, i);

        int from = this.start;
        int to = calculateStart(n, m, i + 1);
        this.interval = new Interval(from, to);

        this.node = nodes.stream().filter(e -> {
            return e >= start;
        }).findFirst().orElse(nodes.get(0));
    }

    public Integer getFrom() {
        return interval.from;
    }

    public Integer getTo(){
        return interval.to;
    }

    public void setInterval(Interval interval) {
        this.interval = interval;
    }

    public Integer getNode() {
        return node;
    }

    public void setNode(Integer node) {
        this.node = node;
    }

    private class Interval{
        private int from;

        private int to;

        private Interval(int from, int to){
            this.from = from;
            this.to = to;
        }
    }

    private static int calculateStart(int n, int m, int i){
        return (int)((n + Math.pow(2, i - 1)) % Math.pow(2, m));
    }

    @Override
    public String toString() {
        return "\tStart : " + start + "\n"
                + "\tInterval : " + interval.from + " " + interval.to + "\n"
                + "\tNode : " + node + "\n";
    }

    public Integer getStart(){
        return start;
    }

}
