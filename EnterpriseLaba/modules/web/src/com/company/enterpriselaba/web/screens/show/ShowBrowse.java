package com.company.enterpriselaba.web.screens.show;

import com.haulmont.cuba.gui.screen.*;
import com.company.enterpriselaba.entity.Show;

@UiController("enterpriselaba_Show.browse")
@UiDescriptor("show-browse.xml")
@LookupComponent("showsTable")
@LoadDataBeforeShow
public class ShowBrowse extends StandardLookup<Show> {
}