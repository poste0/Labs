package com.company.enterpriselaba.service;

import com.company.enterpriselaba.entity.Theatre;
import com.haulmont.cuba.security.entity.User;

import java.util.Date;
import java.util.List;

public interface RegisterService {
    String NAME = "enterpriselaba_RegisterService";

    List<String> register(String firstName, String lastName, String type, Theatre theatre);
}