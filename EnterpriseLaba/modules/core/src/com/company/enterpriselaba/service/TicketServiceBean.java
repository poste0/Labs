package com.company.enterpriselaba.service;

import com.company.enterpriselaba.entity.Ticket;
import com.haulmont.cuba.core.global.DataManager;
import com.haulmont.cuba.security.entity.User;
import org.springframework.stereotype.Service;

import javax.inject.Inject;

@Service(TicketService.NAME)
public class TicketServiceBean implements TicketService {
    @Inject
    private DataManager dataManager;

    @Override
    public void buyTicket(User user, Ticket ticket) {
        ticket.setStatus("Sold");
        dataManager.commit(ticket);
    }
}