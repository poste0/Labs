package com.company.enterpriselaba.service;

import com.company.enterpriselaba.entity.Ticket;
import com.haulmont.cuba.security.entity.User;

public interface TicketService {
    String NAME = "enterpriselaba_TicketService";

    void buyTicket(User user, Ticket ticket);
}