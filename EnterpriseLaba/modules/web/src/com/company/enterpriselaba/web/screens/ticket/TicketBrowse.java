package com.company.enterpriselaba.web.screens.ticket;

import com.haulmont.cuba.gui.screen.*;
import com.company.enterpriselaba.entity.Ticket;

@UiController("enterpriselaba_Ticket.browse")
@UiDescriptor("ticket-browse.xml")
@LookupComponent("ticketsTable")
@LoadDataBeforeShow
public class TicketBrowse extends StandardLookup<Ticket> {
}