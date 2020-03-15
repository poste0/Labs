package ru;

import java.util.ArrayList;
import java.util.List;

public class ChordNode {
    private Integer id;

    private List<FingerTable> fingerStarts;

    private Integer successor;

    private Integer predecessor;

    public ChordNode(int n, int m, List<Integer> nodes){
        this.id = n;

        fingerStarts = new ArrayList<>();
        for (int i = 1; i < m + 1; i++) {
            fingerStarts.add(new FingerTable(n, m, i, nodes));
        }

        successor = fingerStarts.get(0).getNode();

        for(int i = 0; i < nodes.size(); i++){
           if(nodes.get(i) >= n){
               if(i > 0){
                   predecessor = nodes.get(i - 1);
                   break;
               }
               else {
                   predecessor = nodes.get(nodes.size() - 1);
                   break;
               }
           }
        }

    }

    @Override
    public String toString() {
        return fingerStarts.toString() + " " + successor + " " + predecessor;
    }

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public List<FingerTable> getFingerStarts(){
        return fingerStarts;
    }

    public Integer getSuccessor(){
        return successor;
    }

    public Integer getPredecessor(){
        return predecessor;
    }
}
