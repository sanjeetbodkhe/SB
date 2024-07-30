// *****************************************************************************
// ** TableFloaterTitle.js
// **
// ** Unobtrusive JavaScript function which will keep the titles of a table on
// ** screen as the user scrolls down.  The titles will 'float' keeping in view
// ** as long as any part of the table remains in view.
// **
// ** Author: Stan Slaughter
// ** Web: http://www.StanSight.Com/
// *****************************************************************************

  //////////////////////////////////////////////////////////////////////////////
  // startFloatTitle - Start tables floating
  //
  //  This function will scan the page looking for every "table" on it.
  //  Any table which has a class of "FloatTitle" will have its ID value
  //  passed on to the function for the actual float process
  //////////////////////////////////////////////////////////////////////////////

  function startFloatTitle() {

    var cnt=0;
    var allTableIDs = new Array();
    allTables = document.getElementsByTagName("table");

    // Scan page for all tables with a class of "FloatTitle"
    for (i=0;i<allTables.length; i++) {
      if (allTables[i].className == "FloatTitle") {

        // Get the ID. If no ID exists then assign one
        tableID = allTables[i].id;
        if ( tableID == "") {
          tableID = "tmpFloatTitleTableId" + cnt;
          allTables[i].id = tableID;
        } 

        // Save ID's
        allTableIDs[cnt] = tableID;
        cnt++;
      }
    }

    // Start floating the tables title
    for (i=0;i<allTableIDs.length; i++) {
      floatTitle(allTableIDs[i],i);
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // floatTitle - Start the float process
  //  @param tableID - ID of table that floating title will be created for
  //  @param cnt - Number representing table - for tracking seperate float events
  //
  //  This function calls the functions which create the floating title and
  //  start the floating process
  //////////////////////////////////////////////////////////////////////////////

  var floatTitleTimer = new Array();
  function floatTitle(tableID,cnt) {

    // Stop any old Loops
    if (typeof(floatTitleTimer[cnt])== 'undefined') floatTitleTimer[cnt] = 0;
    clearTimeout(floatTitleTimer[cnt]);

    // Create title object then start float loop
    titleID = createTitleObj(tableID);
    floatTitleLoop(tableID,titleID,cnt);

  }

  //////////////////////////////////////////////////////////////////////////////
  // createTitleObj - Create the floating object for this table
  //  @param tableID - ID of table that floating title will be created for
  //  @return titleWrapperID - ID of div which contains a copy of tables THEAD
  //
  //  Clone the THEAD from the table, create a new table to place it in then
  //  create a DIV wrapper to place it in. The ID for the DIV is returned to
  //  the calling function so it can be used to position the object in the
  //  floating function.
  //////////////////////////////////////////////////////////////////////////////

  function createTitleObj (tableID) {

    var titleWrapperID = tableID + "Title";
    var titleTableID = tableID + "TitleTable";

    // If Title has not been created yet
    if (document.getElementById(titleWrapperID)==null) {

      // "clone" the Table's thead
      tableObj = document.getElementById(tableID);
      clonedtHead=tableObj.tHead.cloneNode(true);

      // Create wrapper div
      titleWrapperObj = document.createElement("div");
      titleWrapperObj.setAttribute("id", titleWrapperID);

      // Create the Title table
      TitleTable = document.createElement("table");
      TitleTable.setAttribute("id", titleTableID);
      titleTableIDNew=titleTableID;

      // Append Cloned thead to the Title table
      TitleTable.appendChild( clonedtHead );

      // Append table to div container
      titleWrapperObj.appendChild(TitleTable);

      // Insert wrapper div into the DOM before Table
      parentDiv = tableObj.parentNode;
      parentDiv.insertBefore(titleWrapperObj, tableObj);

      // Initially position container
      titleWrapperObj.style.position="absolute";
      titleWrapperObj.style.top = "0px";
      titleWrapperObj.style.zIndex="1";
    }

    tableObj = document.getElementById(tableID);
    titleTableObj = document.getElementById(titleTableID);

    // Set Title width to be the same as Table
    titleWrapperObj = document.getElementById(titleWrapperID);
    titleWrapperObj.style.width = tableObj.offsetWidth + 'px';

    // Set widths of Title TD's to same as Table TD's
    tableCells = tableObj.tHead.rows[0].cells;
    titleTableCells = titleTableObj.tHead.rows[0].cells;

    for (var i=0; i != tableCells.length; i++) {
      titleTableCells[i].style.width = (tableCells[i].clientWidth)+ 'px';
      titleTableCells[i].style.cursor = 'normal';
    }

    return titleWrapperID;
  }

  //////////////////////////////////////////////////////////////////////////////
  // floatTitleLoop - Float the titles up and down the table as the page scrolls
  //  @param tableID - ID of table that wants floating title
  //  @param titleID - ID of div that will float (contains copy of tableID's THEAD)
  //  @param cnt - Number representing table - for tracking seperate float events
  //
  //  Keep the title object in view as the table postion changes as a user
  //  scrolls up and down in the window
  //////////////////////////////////////////////////////////////////////////////

  function floatTitleLoop(tableID,titleID,cnt) {

    // If Table and Title objects exist
    if (document.getElementById(tableID) !=null && document.getElementById(titleID) !=null) {

      // Set value to be number of pixels from screen top that you wish
      // scrolling to start at. 0=top, 10=10 pixels down from top, etc..
      // Useful for those who happen to have top screen banners
      var	startOffsetTop = 0;

      // Get start and stop float positions from table
      var tableObj = document.getElementById(tableID);
      var tablePos = FindPosition(tableObj);
      var startTop = tablePos[1] - startOffsetTop;
      var endTop = startTop + tableObj.clientHeight;		

      // Get new positon of body scroll top and keep it in bounds
      var newTop =(document.documentElement.scrollTop>0) ? document.documentElement.scrollTop : document.body.scrollTop;
      if (newTop < startTop) newTop = startTop;
      if (newTop > endTop) newTop = endTop;

      // Move the title to new postion
      var titleObj = document.getElementById(titleID);
      if (newTop > startTop && newTop < endTop) {

        // Display "Title" if it is not all ready being displayed
        if (titleObj.style.display != 'block') titleObj.style.display='block';

        // Apply offset to get newTop position
        newTop = newTop + startOffsetTop;

        // Apply Ease-In effect
        easeInWanted=true;
        if (easeInWanted) {

          // Get current location and get the difference from where it's
          // at to where you want it to go
          oldTop = parseInt(titleObj.style.top);
          topDiff=newTop-oldTop; 

          // Calaculate how far you want to move it - then set that to new postion
          moveTop=0;
          if ( (topDiff < (-1) ) || (topDiff > (1) ) ) { moveTop=Math.round(topDiff/3);}
          newTop = oldTop + moveTop;
        }

      	// Move to new top position
      	if (newTop < 0) newTop = 0;
      	titleObj.style.top = newTop + "px";
      } else {
        // Else hide "Title" if it is not all ready hidden
        if (titleObj.style.display != 'none') {
          titleObj.style.display='none';
          titleObj.style.top = "0px";
          titleObj.style.zIndex="1";
        }
      }

    }

    // Execute this function again in 40 milliseconds (thousandths of a second)
    cmd = "floatTitleLoop('" + tableID + "','" + titleID + "','" + cnt + "')";
    floatTitleTimer[cnt] = window.setTimeout(cmd, 40); 

  }

  //////////////////////////////////////////////////////////////////////////////
  // FindPosition - find the Top and Left postion of an object on the page
  //  @param obj - object of element whose position needs to be found
  //  @return array - Array whoose first eleemnt is the left postion and whoose
  //                  second is the top position
  //////////////////////////////////////////////////////////////////////////////

  function FindPosition(obj) {
    // Figure out where the obj object is in the page by adding
    // up all the offsets for all the containing parent objects
    if (obj == null) return [0,0];

    // Assign the obj object to a temp variable
    tmpObj = obj;

    // Get the offsets for the current object
    var obj_left = tmpObj.offsetLeft;
    var obj_top = tmpObj.offsetTop;

    // If the current object has a parent (ie contained in a table, div, etc..)
    if (tmpObj.offsetParent) {

      // Loop through all the parents and add up their offsets
      // The while loop will end when no more parents exist and a null is returned
      while (tmpObj = tmpObj.offsetParent) {
      	obj_left += tmpObj.offsetLeft;
      	obj_top += tmpObj.offsetTop;
      }
    }
    return [obj_left , obj_top];
  }

   function printpr() {
     var OLECMDID = 7;
     /* OLECMDID values:
      * 6 - print
      * 7 - print preview
      * 1 - open window
      * 4 - Save As
      */

     var t  = document.getElementById(titleTableIDNew);
     t.style.display = 'none';

     var PROMPT = 1; // 2 DONTPROMPTUSER
     var WebBrowser = '<OBJECT ID="WebBrowser1" WIDTH=0 HEIGHT=0 CLASSID="CLSID:8856F961-340A-11D0-A96B-00C04FD705A2"></OBJECT>';
     document.body.insertAdjacentHTML('beforeEnd', WebBrowser);
     WebBrowser1.ExecWB(OLECMDID, PROMPT);
     WebBrowser1.outerHTML = "";
     t.style.display='block';
  }

   function printScreen() {

     var t  = document.getElementById(titleTableIDNew);
     t.style.display = 'none';
     window.print();
  }
  //////////////////////////////////////////////////////////////////////////////
  // Start floating titles after window finishes loading
  ////////////////////////////////////////////////////////////////////////////// 

  var titleTableIDNew;
  window.onload = startFloatTitle;

  //////////////////////////////////////////////////////////////////////////////
  // Start floating titles when window is resized
  ////////////////////////////////////////////////////////////////////////////// 

  window.onresize = startFloatTitle;