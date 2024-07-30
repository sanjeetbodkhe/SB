var leafHandlerLoaded = void 0;
var rootHandlerLoaded = void 0;
$(document).ready(function(){
        /* SD 2012-03-13 The dropdowns had an onchange event
         *  that redirected the page to selected value. If you hit
         *  the browser back button the page went back to the original
         *  page, except for the dropdown that stayed the same. This
         *  was because the browser caches the form values; particularly IE.
         *  In order to avoid caching of the form elements we shall..
         *  1. remove the onchange attributes on the dropdown
         *  2. transfer the onchange to a JQuery onchange event
         *  3. When an onchange is triggered, capture the new selected value
         *  4. Revert the dropdowns to the original values, then
         *  5. redirect the page to the newly selected values.
         *  This is an ugly solution. It could have been put in the 
         *  meta tags in the original html files.
         */
         if($('select[name="Drop1"]').length > 0)
         {
             /*
              *SD 2012-03-13 The following 5 lines may not be necessary, but they do make the html valid-html.
              */
            var dropdown = $('select[name="Drop1"]').html();
            dropdown = dropdown.replace('SELECTED=""','selected="selected"');
            dropdown = dropdown.replace('SELECTED','selected="selected"');
            dropdown = dropdown.replace('selected=""','selected="selected"');
            $('select[name="Drop1"]').html(dropdown);

            //Drop1
            $('select[name="Drop1"]').removeAttr('onchange');
            var originalSelection = $('select[name="Drop1"] option:selected').attr('value');
            $('select[name="Drop1"]').change(function(){
                var currentSelection = $(this).attr('value');
                $('select[name="Drop1"]').find('option[value="'+originalSelection+'"]').attr('selected','selected');
                window.location = currentSelection;
                return false;
             });
         }

         //DropF
         if($('select[name="DropF"]').length > 0)
         {
            $('select[name="DropF"]').removeAttr('onchange');
            var originalSelectionF = $('select[name="DropF"] option:selected').attr('value');
            $('select[name="DropF"]').change(function(){
                var currentSelectionF = $(this).attr('value');
                $('select[name="DropF"]').find('option[value="'+originalSelectionF+'"]').attr('selected','selected');
                window.location = currentSelectionF;
                return false;
             });
         }
        });
(function(){
	/**
* @preserve  EIA Highcharts Theme v1.1.5 2014-04-10
*
* Created by Shivan Computers Corporation on behalf 
* of the U.S. Energy Information Administration:
* Authors : Ryan Lynch (Ryan.Lynch@eia.gov) and Scott Gearhart (Scott.Gearhart@eia.gov)
**/
!function(e){var t,o,r,s,n,a,l=e.Chart,p=(e.StockChart,e.extend),h=e.each,c=e.map,u=e.Axis,g=e.pick,x=e.PREFIX,f=e.Tick,b=Math,v=(b.round,b.floor),T=(b.ceil,b.max),A=(b.min,b.abs,b.cos,b.sin,b.PI,function(e){return e!==t&&null!==e}),C=e.splat||function(e){return $.isArray(e)||(e=[e]),e},w=e.merge,k=Math.max,S=Math.min,_=(Math.pow,Math.abs,document),L=window,O=(function(){function e(){}}(),null!=navigator.userAgent.match(/MSIE/));function M(e,t,i){return $(t.chart.container).width()<300&&!(t.chart.width&&t.chart.width>=300)&&(t.exporting.buttons.contextButton.text=null),t}function P(t){var i,o=this,r=(o.options,new e.Series);r.init(o.dummyChart,$.extend(!0,{},t)),o.dummyChart.series.push(r),i=R.call(o.dummyChart,r),t.xType=r.options.xType=void 0!==t.xType?t.xType:i.xType,t.yType=r.options.yType=void 0!==t.yType?t.yType:i.yType}function D(t){var i=this;if(i.dummyChart.series){var o=$.isArray(t.yAxis)?t.yAxis:[t.yAxis];h(o,(function(e){e.yType=null})),h(i.dummyChart.series,(function(t){var i=void 0!==t.options.yAxis?t.options.yAxis:0;o[i].yType=null==o[i].yType||t.options.yType==o[i].yType?t.options.yType:e.yTypes.mixed})),h(o,(function(t,o){switch(t.yType){case e.yTypes.negative:t.min=void 0!==t.min?t.min:null,t.max=void 0!==t.max?t.max:0;break;case e.yTypes.positive:t.min=void 0!==t.min?t.min:0,t.max=void 0!==t.max?t.max:null;break;case e.yTypes.mixed:default:t.min=void 0!==t.min?t.min:null,t.max=void 0!==t.max?t.max:null}$.isArray(i.yAxis)&&(i.yAxis[o].min=i.yAxis[o].options.min=t.min,i.yAxis[o].max=i.yAxis[o].options.max=t.max)}))}switch(t.minXType=j.call(i.dummyChart,S),t.maxXType=j.call(i.dummyChart,k),xAxis=C(t.xAxis),xAxis[0].minRange=void 0!==xAxis[0].minRange?xAxis[0].minRange:e.xMinRanges[t.minXType],t.minXType){case e.xTypes.annual:xAxis[0].minTickInterval=31536e6;break;case e.xTypes.monthly:xAxis[0].minTickInterval=24192e5;break;case e.xTypes.weekly:xAxis[0].minTickInterval=6048e5;break;case e.xTypes.daily:xAxis[0].minTickInterval=864e5;break;case e.xTypes.quarterly:xAxis[0].minTickInterval=7776e6;break;case e.xTypes.hourly:xAxis[0].minTickInterval=36e5}}function E(){this.seriesColorQueue=Highcharts.getOptions().colors.slice(0)}function F(){return 0==this.seriesColorQueue.length&&E.apply(this),this.seriesColorQueue.shift()}function H(e){if(void 0!==e.lastHistoricalPeriod){var t=function(e){return 4==e.length&&(e+="01"),6==e.length&&(e+="01"),8==e.length&&(e+="00"),e.indexOf("Q")>=0&&(e=e.substr(0,e.indexOf("Q"))+("00"+(3*e[e.indexOf("Q")+1]-2)).slice(-2)+e.substr(e.indexOf("Q")+2,e.length-e.indexOf("Q")+2)),e}(String(e.lastHistoricalPeriod)),i=e.data.length;e.data.reverse();var o=$.extend(!0,{},e,{linkedTo:":previous",dashStyle:"Dash",projection:!0,id:e.id?e.id+"-projection":null});for(delete o.lastHistoricalPeriod,o.data=[];i--;){var r=Array.isArray(e.data[i])?e.data[i][0]:e.data[i].x,s=new Date(r).toISOString().replace(/-/g,"").replace(/:/g,"").replace(/T/g,"").substr(0,10);s>=t&&(o.data.push(e.data[i]),s>t&&e.data.splice(i,1))}return e.data.reverse(),P.call(this,o),o}return null}function U(e){var t=0,i=0,o=0;switch(e.xType){case 4:i=7;break;case 3:t=1;break;case 2:t=3;break;case 1:o=1}for(var r=void 0,s=void 0,n=!1,a=0;a<e.data.length;a++){var l=new Date(Array.isArray(e.data[a])?e.data[a][0]:e.data[a].x);if(void 0!==r){m=l.getUTCMonth(),d=l.getUTCDate(),y=l.getUTCFullYear(),mLast=r.getUTCMonth(),dLast=r.getUTCDate(),yLast=r.getUTCFullYear();var p=Date.UTC(y,m,d),h=Date.UTC(yLast+o,mLast+t,dLast+i);h!==p&&(null==s&&(e.data[a-1]={marker:{enabled:!0,lineWidth:1},y:Array.isArray(e.data[a-1])?e.data[a-1][1]:e.data[a-1].y,x:Array.isArray(e.data[a-1])?e.data[a-1][0]:e.data[a-1].x},n|=!0),e.data.splice(a++,0,{x:h,y:null,export:!1}),r=null)}s=r,r=l}r&&!s&&(e.data[e.data.length-1]={marker:{enabled:!0,lineWidth:1},y:e.data[e.data.length-1][1],x:e.data[e.data.length-1][0]},n|=!0),n&&(e.turboThreshold=0)}function I(t,i,o){$(i.chart.container).width()<300&&!(i.chart.width&&i.chart.width>=300)&&(i.exporting.buttons.contextButton.text=null);var r=function(t){var i,o,r=[];if(t.yAxis instanceof Array){if(t.yAxis.length>1){for(i=0;i<t.yAxis.length;i++){for(r[i]=[],o=0;o<t.series.length;o++)(t.series[o].yAxis&&t.series[o].yAxis==i||!t.series[o].yAxis&&0==i)&&r[i].push(o);t.yAxis.length>2&&(void 0===t.yAxis[i].title&&(t.yAxis[i].title={}),t.yAxis[i].title.align="middle")}for(i=0;i<r.length;i++){var s=null,n=!0;for(o=0;o<r[i].length;o++){var a=r[i][o];null==s&&(s=void 0===t.series[a].color?e.eiaTheme.colors[a]:t.series[a].color),s!=(void 0===t.series[a].color?e.eiaTheme.colors[a]:t.series[a].color)&&(n=!1)}n&&$.extend(!0,t.yAxis[i],{title:{style:{color:s}},labels:{style:{color:s}}})}}if(t.series.length>1&&t.userGenerated&&($.extend(!0,t,{chart:{logo:Highcharts.logos.none}}),t.credits.text==e.eiaTheme.credits.text&&(t.credits.text=t.credits.text.replace("Source","Data source"))),t.yAxis.length>1&&t.yAxis.length<=4)t.yAxis.length>2&&$.extend(!0,t,{captions:{items:[{text:"Warning:  Please take care in interpreting this chart. You have selected series with different units resulting in a chart with "+t.yAxis.length+" axes.",style:{color:"red"}}]}});else if(t.yAxis.length>4)return!1}return!0}(i);if(!r)return i.lang.noData="<div align='center'>You have chosen data series resulting in more than 4 axes.<br> Series with the same units will be plotted on the same axis. Axes will be added for each unique unit.<br> You must select series that result in 4 or fewer axes.</div>",i.noData.style={fontSize:"14px",fontWeight:"bold",color:"red"},i.series=[],i;var s,n=this,a=i.chart.renderTo,l="string"==typeof a?$("#"+a):$(a),p=0,h=$.isArray(i.yAxis)?i.yAxis:[i.yAxis],c=$("<div></div>").appendTo("body").css({position:"absolute",top:-9999}),d=new e.Renderer(c[0],0,0);if(n.dummyChart={options:i,counters:{wrapColor:function(e){this.color>=e&&(this.color=0)},wrapSymbol:function(e){this.symbol>=e&&(this.symbol=0)},color:0,symbol:0},series:[],xAxis:[],yAxis:[]},i.xAxis instanceof Array)for(u=0;u<i.xAxis.length;u++)i.xAxis[u].index=u,n.dummyChart.xAxis.push({options:i.xAxis[u],series:[]});else i.xAxis.index=0,n.dummyChart.xAxis.push({options:i.xAxis,series:[]});if(i.yAxis instanceof Array)for(u=0;u<i.yAxis.length;u++)i.yAxis[u].index=u,n.dummyChart.yAxis.push({options:i.yAxis[u],series:[]});else i.yAxis.index=0,n.dummyChart.yAxis.push({options:i.yAxis,series:[]});if(!i.chart.forExport){if(i.series)for(var u=0;u<i.series.length;u++){i.series[u].color||(i.series[u].color=F.apply(n));var g=H.call(n,i.series[u]);P.call(n,i.series[u]),i.series[u].xType<=4&&i.series[u].xType>=1&&i.chart.breakLines&&U(i.series[u]),g&&(i.series.splice(u+1,0,g),u++)}D.call(n,i),$.support.touch?i.chart.zoomType="none":i.chart.zoomType=void 0!==i.chart.zoomType?i.chart.zoomType:X.call(n.dummyChart)}for(i.tooltip.shared&&(i.tooltip.crosshairs=i.tooltip.shared),u=0;u<h.length;u++)p=Math.max(p,N.call(n,h[u],d)),"logarithmic"==h[u].type&&(h[u].min=void 0!==h.min?h.min:null);return s=z.call(n,i.title,d,(i.chart.width||$(l).width()||600)-(!1!==i.exporting.enabled?44+(null==i.exporting.buttons.contextButton.text?0:100):0)-(i.chart.spacing[1]+i.chart.spacing[3])),c.remove(),i.chart.spacingTopAdjust=p,i.chart.titleAdjust=s,s>0?i.subtitle.y=(isNaN(i.subtitle.y)?0:i.subtitle.y)+s+(isNaN(i.chart.spacingTop)?i.chart.spacing[0]:i.chart.spacingTop)+5:i.chart.spacing[0]=i.chart.spacingTop=(isNaN(i.chart.spacingTop)?i.chart.spacing[0]:i.chart.spacingTop)+p,i}function N(t,i){var r=this,n=jQuery.extend(!0,{},o,t).title,a=0,l=0,p=n.originalAlign||(void 0===n.align?"above":n.align.toLowerCase()),h=0;if("above"==p){var c=i.text(n.text,0,0).attr({rotation:0}).css(jQuery.extend({},r.dummyChart.options.chart.style)).css(n.style).add().getBBox();"above"==n.align&&(r.dummyChart.options.chart&&r.dummyChart.options.chart.type&&"bar"==r.dummyChart.options.chart.type?jQuery.extend(!0,t,{title:{align:"low"}}):jQuery.extend(!0,t,{title:{offset:-1*c.width,rotation:0,align:"high",x:t.opposite?c.width:0,y:-1*(c.height+c.y+10)}})),h=c.height}return jQuery.extend(!0,t,{title:{originalAlign:p}}),t.labels&&t.labels.formatter||jQuery.extend(!0,t,{labels:{formatter:function(){var t=this.value.toString(),i=/\d+\.?(\d*)/;return i.test(t)&&(l=Math.max(this.value.toString().match(i)[1].length,l),void 0===a&&(a=l),this.isLast?(l!=a&&(a=l,s=!0),l=0,t=e.numberFormat(this.value,a,".",",")):t=e.numberFormat(this.value,a,".",",")),t}}}),h}function z(e,t,o){var r,s=0;if(e&&e.text){e.text=e.text.replace(/<br\/?>/gi," ");e.style&&void 0!==e.style.width&&parseInt(e.style.width);for(r=l.prototype.splitText(e.text,e.style,o,t),e.text=$.map(r,(function(e){return e.line})).join("<br/>"),i=0;i<r.length;i++)s+=r[i].lineHeight}return s}function R(t){var i,o,r,s,n,a,l,p,h,c,d,u,g,y,x="datetime"==C(this.options.xAxis)[0].type,m=!1,f=!1;i=o=r=s=!0,n=a=l=p=null,threeMonth=!0;var b=t.xData||t.data,v=t.yData||t.data;for(g=0;g<b.length;g++){x&&(h=(y=new Date(void 0!==b[g]&&void 0!==b[g].x?b[g].x:b[g])).getUTCMonth(),c=y.getUTCDate(),d=y.getUTCDay(),u=y.getUTCHours(),null!==n&&null!==a&&null!==l&&null!==p&&(i&=h==n,r&=d==l,s&=u==p,(o&=c==a)&&s&&!i&&((h-n)%3==0?threeMonth&=!0:threeMonth=!1)),n=h,a=c,l=d,p=u);var T=v[g]&&void 0!==v[g].y?v[g].y:v[g];m|=T<0,f|=T>0}return{xType:x?i&&o&&s?e.xTypes.annual:o&&s&&threeMonth?e.xTypes.quarterly:o&&s?e.xTypes.monthly:r&&s?e.xTypes.weekly:s?e.xTypes.daily:e.xTypes.hourly:e.xTypes.category,yType:m&&f||"value"==t.options.compare||"percent"==t.options.compare?e.yTypes.mixed:m?e.yTypes.negative:e.yTypes.positive}}function j(e){var t,i,o;for(t=0;t<this.series.length;t++)o=this.series[t].options.xType,i=isNaN(i)?o:e(i,o);return i}function X(){var e,t,i=!1;for(e=0;e<this.series.length&&(t=this.series[e],!(i|=/column|bar/.test(t.type)));e++);return i?"":"xy"}function Y(){var e=this,t=e.options.chart.printContainer,i=[],o=t.parentNode,r=_.body,s=r.childNodes;e.isPrinting||(e.isPrinting=!0,$(t).width(e.chartWidth).height(e.chartHeight),h(s,(function(e,t){1===e.nodeType&&(i[t]=e.style.display,e.style.display="none")})),r.appendChild(t),e.buttonGroup.hide(),L.focus(),L.print(),setTimeout((function(){o.appendChild(t),e.buttonGroup.show(),h(s,(function(e,t){1===e.nodeType&&(e.style.display=i[t])})),$(t).width("auto").height("auto"),e.isPrinting=!1}),1e3))}function W(){var t,o,r,s=this;for(i=0;i<s.series.length;i++)(t=s.series[i]).options.hasOwnProperty("xType")||(o=R.call(s,t),t.options.xType=o.xType,o.yType===e.yTypes.mixed&&(r=t.yAxis.getExtremes(),t.yAxis.setExtremes(r.min?r.min:null,r.max,!1)));V.apply(s)}function B(){var e=this;e.options.originalSeriesOptions||(e.options.originalSeriesOptions=c(e.series,(function(e){return jQuery.extend({color:e.color},e.options)}))),e.options.originalColorCounter=this.counters.color,V.apply(e)}function V(){var e=this,t=!!e.options.chart.lineUpYAxisZeros&&Q.apply(e);s&&(s=!1,t||G.apply(e))}function Q(){var e,t,o,r,s=this,n=!1,a=[],l=!0,p=!0,h=void 0,c={},d=s.yAxis;if(this.axisZerosAligned++,d.length>1){for(i=0;i<s.yAxis.length;i++)r=d[i].getExtremes(),a[i]={min:e=Math.abs(r.min),max:t=Math.abs(r.max),axisLength:o=t+e,positiveRatio:Math.abs(t)/o,negativeRatio:Math.abs(e)/o};for(i=0;i<a.length;i++){var u=a[i].positiveRatio,g=a[i].negativeRatio,y=Math.abs(u-g);isNaN(u)||isNaN(g)||(l&=1==u,p&=1==g,(void 0===h||y<h)&&(h=y,c={positiveRatio:u,negativeRatio:g}))}for(1!=h||l||p||(c={positiveRatio:.5,negativeRatio:.5}),i=0;i<s.yAxis.length;i++)if(a[i].positiveRatio!=c.positiveRatio&&!isNaN(a[i].min)&&!isNaN(a[i].max)){n=!0,c.positiveRatio<a[i].positiveRatio?d[i].setExtremes(a[i].max*c.negativeRatio/c.positiveRatio*-1,a[i].max,!1):d[i].setExtremes(-1*a[i].min,a[i].min*c.positiveRatio/c.negativeRatio,!1)}}return n&&G.apply(s),this.axisZerosAligned--,n}function Z(){var e=this;if(e.options.chart&&e.options.chart.lineUpYAxisZeros)for(var t=0;t<e.yAxis.length;t++){var i=e.yAxis[t];null==i.userMin&&null==i.userMax||i.setExtremes(null,null,!1)}}function G(){var e,t=this,i=$.isArray(t.yAxis)?t.yAxis:[t.yAxis];for(e=0;e<i.length;e++)i[e].isDirty=!0;for(e=0;e<t.series.length;e++)t.series[e].isDirty=!0;null!==t.redraw?t.redraw(!1):t.detachedredraw(!1)}p(e,{eia_theme_version:"1.1.3",logos:{none:0,eia:1,reuters:2,bloomberg:3},eia_blue:"#0096d7",eia_tan:"#bd732a",eia_green:"#5D9732",eia_yellow:"#ffc702",eia_red:"#a33340",eia_brown:"#403203",eia_lt_blue:"#76d5ff",eia_lt_green:"#bed5ad",eia_dk_red:"#410e14",eia_grey:"#666666",eia_dk_blue:"#003953",eia_dk_green:"#2a4b11",eia_dk_grey:"#333333",eia_mut_purple:"#746395",eia_mut_red:"#b55c66",eia_mut_blue:"#66a0b9",eia_mut_brown:"#a27d33",eia_mut_green:"#92a753",eia_mut_blue_sub_sh1:"#99c0d0",eia_mut_blue_sub_sh2:"#66a0b9",eia_mut_blue_sub_sh3:"#3380a2",eia_mut_blue_sub_sh4:"#003953",eia_projections_line:{color:"#888",dashStyle:"Dash",width:1,zIndex:3},eia_projections_label:{color:"#888",fontWeight:"bold"},eia_zero_axis_line:{color:"#333",dashStyle:"Solid",zIndex:1,width:1,value:0}}),p(e,{eiaTheme:{chart:{animation:!O,plotBackgroundColor:"rgba(255, 255, 255, .1)",spacingBottom:20,logo:e.logos.eia,events:{redraw:function(){W.apply(this,arguments)},load:function(){B.apply(this,arguments)}},style:{fontFamily:"Arial, Verdana, Helvetica, sans-serif"},lineUpYAxisZeros:!1,borderColor:"#ffffff",breakLines:!0},colors:[e.eia_blue,e.eia_tan,e.eia_green,e.eia_yellow,e.eia_red,e.eia_brown,e.eia_lt_blue,e.eia_lt_green,e.eia_dk_red,e.eia_grey,e.eia_dk_blue,e.eia_dk_green,e.eia_dk_grey],title:{align:"left",margin:35,style:{color:"#333",fontSize:"16px",fontWeight:"bold",fontFamily:"Arial, Verdana, Helvetica, sans-serif",whiteSpace:"nowrap"}},subtitle:{style:{color:"#333",fontSize:"12px",fontFamily:"Arial, Verdana, Helvetica, sans-serif"},align:"left",floating:!1},credits:{style:{color:"#888",fontSize:"11px"},text:"Data source: U.S. Energy Information Administration",href:"http://www.eia.gov",position:{align:"left",verticalAlign:"bottom",x:15}},labels:{style:{fontSize:"12px",color:"#333"}},xAxis:{labels:{style:{color:"#333"}},endOnTick:!1,startOnTick:!1,tickColor:"#333",lineColor:"#333",title:{style:{color:"#888",fontWeight:"bold",fontSize:"12px",fontFamily:"Arial, Verdana, Helvetica, sans-serif"}},dateTimeLabelFormats:{day:"%e %b. %Y",week:"%e %b. %Y",month:"%b '%y",year:"%Y"}},yAxis:{labels:{style:{color:"#333",fontFamily:"Arial, Verdana, Helvetica, sans-serif"}},lineColor:"#A0A0A0",minorTickInterval:null,tickColor:"#A0A0A0",tickColor:"#A0A0A0",tickWidth:0,title:{align:"above",style:{color:"#888",fontWeight:"bold",fontSize:"12px"}}},tooltip:{backgroundColor:"rgba(255, 255, 255, 0.75)",style:{fontSize:"12px",color:"#333333",padding:5},formatter:function(){return e.universalTooltipFormatter.apply(this,arguments)}},plotOptions:{series:{animation:!O,tooltip:{showName:!0}},line:{shadow:0,lineWidth:2,borderWidth:0,dataLabels:{color:"#333"},marker:{enabled:!1,states:{hover:{enabled:!0,radius:5}}}},area:{borderWidth:0,shadow:!1,lineWidth:0,marker:{enabled:!1,states:{hover:{enabled:!0,radius:4}}}},spline:{marker:{lineColor:"#333"}},column:{borderWidth:0,shadow:!1,lineWidth:0},bar:{shadow:!1},pie:{allowPointSelect:!0,shadow:!1,dataLabels:{style:{fontSize:"12px",color:"red"}}},scatter:{marker:{radius:3,symbol:"circle",states:{hover:{enabled:!0,lineColor:"rgb(100,100,100)"}}},states:{hover:{marker:{radius:3,enabled:!0}}}}},legend:{floating:!1,borderWidth:1,borderColor:"#e4e4e4",backgroundColor:"#f1f1f1",borderRadius:0,symbolPadding:5,itemStyle:{textDecoration:"none"},itemHoverStyle:{color:"#189bd7",textDecoration:"underline"},itemHiddenStyle:{color:"#CCC"}},lang:{exportButtonTitle:"Export an image or data"},exporting:{buttons:{contextButton:{symbol:"download",symbolFill:"#666",symbolStroke:"#666",symbolStrokeWidth:1,symbolSize:16,symbolX:8,symbolY:8.5,text:"DOWNLOAD",menuItems:[{text:"Print Chart",onclick:function(){Y.apply(this)}},{separator:!0},{text:"Download Image",onclick:function(){this.exportChart({url:"/global/scripts/jquery/highcharts/exporting-server/index.php",width:this.chartWidth},e.generateEIAExportOptions.apply(this))}},{text:"Download PDF",onclick:function(){this.exportChart({url:"/global/scripts/jquery/highcharts/exporting-server/index.php",type:"application/pdf"},e.generateEIAExportOptions.apply(this))}},{text:"Download Data",onclick:function(){this.generateCSV()}},null]}},combineLinkedSeriesInExport:!0}},yTypes:{mixed:0,positive:1,negative:-1},xTypes:{category:0,annual:1,quarterly:2,monthly:3,weekly:4,daily:5,hourly:6},csvXLabels:["Category","Year","Quarter","Month","Week of","Day"],tooltipXLabels:["","Year","Quarter","Month","Week of","Day"],xMinRanges:[0,126144e6,31536e6,107136e5,24192e5,6048e5,864e5],parseXValue:function(t,i,o){var r;switch(t){case e.xTypes.annual:r=e.dateFormat("%Y",i);break;case e.xTypes.monthly:r=e.dateFormat("%b %Y",i);break;case e.xTypes.weekly:case e.xTypes.daily:r=e.dateFormat("%m/%e/%Y",i);break;case e.xTypes.quarterly:m=e.dateFormat("%m",i),y=e.dateFormat("%Y",i),r="Q"+Math.ceil(m/3)+" "+y;break;case e.xTypes.hourly:r=e.dateFormat("%m/%e/%Y %HH",i)+(void 0!==o?" "+o:"");break;default:r=i}return r},universalTooltipFormatter:function(){var t="";if(null!=this.point){var i=this.series,o=this.series.chart,r=i.options.tooltip,s=o.options,n=(!(g="pie"==s.chart.type)&&s.xAxis.type,i.name),a=i.options.xType,l=g?"":i.xAxis.options.categoryTitle||i.xAxis.options.title.text||e.tooltipXLabels[a]||"",p=g?"":i.yAxis.options.title.text||"";g?(c=this.point.name,h=l,y=this.y):(c=n,h=(l.length>0?l+" : ":"")+e.parseXValue(a,this.point.options&&void 0!==this.point.options.tooltipX?this.point.options.tooltipX:this.x,o.userOptions.timezone),this.series.userOptions&&this.series.userOptions.projection&&this.series.data.indexOf(this.point)>0&&(h+=" (Projected)"),y=null!=i.options.compare?e.numberFormat(this.point.change,r.precision||void 0)+" "+p:e.numberFormat(this.y,r.precision||void 0)+" "+p),t=(i.options.tooltip.showName?"<b>"+c+"</b><br/>":"")+h+(""!=h?"<br/>":"")+y}else{var h,c,d,u,g,y="",x=this.points;!(g="pie"==(o=x[0].series.chart).options.chart.type)&&o.options.xAxis.type,a=o.options.maxXType,l=e.tooltipXLabels[a];if(g)for(d=0;d<x.length;d++)t+="<b>"+(c=(u=x[d]).point.name)+"</b> "+(y=e.numberFormat(u.y,void 0)),d<x.length&&(t+="</br>");else for(t+=(h=(""!=l?l+" : ":"")+e.parseXValue(a,this.x,o.userOptions.timezone))+"<br/>",d=0;d<x.length;d++)u=x[d],r=(i=x[d].series).options.tooltip,c=n=i.name,p=i.yAxis.options.title.text||"",t+="<b>"+c+"</b> "+(y=e.numberFormat(u.y,r.precision||void 0)+" "+p),d<x.length&&(t+="<br/>")}return t},generateEIAExportOptions:function(){return{}},numberFormat:function(t,i,o,r){if(null==t||""===t||isNaN(t)||Math.abs(t)===1/0)return"";var s=e.getOptions().lang,n=t,a=isNaN(i=Math.abs(i))?Math.min(n.toString().match(/\d+\.?(\d*)/)[1].length,20):i,l=void 0===o?s.decimalPoint:o,p=void 0===r?s.thousandsSep:r,h=n<0?"-":"",c=String(parseInt(n=Math.abs(+n||0).toFixed(a))),d=c.length>3?c.length%3:0;return h+(d?c.substr(0,d)+p:"")+c.substr(d).replace(/(\d{3})(?=\d)/g,"$1"+p)+(a?l+Math.abs(n-c).toFixed(a).slice(2):"")}}),e.setOptions($.extend(!0,e.eiaTheme,{exporting:{csvOptions:{overrideURL:null,postURL:"/global/scripts/jquery/highcharts/exporting-server/csv_exporter.php"}}})),p(l.prototype,{resetZoom:function(){var e=this.xAxis;h(e,(function(e){e.setExtremes(null,null,!1)}))},resetSeries:function(e){var t,i=this;for(void 0===e&&(e=!0),i.removeAllSeries(e),t=0;t<i.options.originalSeriesOptions.length;t++)i.addSeries(i.options.originalSeriesOptions[t],!1);i.counters.color=i.options.originalColorCounter,e&&i.redraw()},removeAllSeries:function(e){var t,i=this,o=[];for(void 0===e&&(e=!0),t=0;t<i.series.length;t++)i.scroller&&i.scroller.series===i.series[t]||o.push(i.series[t]);for(t=0;t<o.length;t++)o[t].remove(!1);i.scroller&&i.scroller.series&&(i.scroller.series=null),i.counters.color=0,E.apply(i),e&&i.redraw()},splitText:function(e,t,i,o){var r,s,n,a,l,p,h=[],c=0,d=0;for(words=e.match(/(<\s*(\w+)[^>]*>.*?<\s*\/\2\s*>|<\s*\/[^>]+>|[^<>\s]+|(<[^>]+>))/g).concat("");d<words.length;){d=c,l="";do{if(r&&(s=r.getBBox().height,r.destroy()),l+=(n=words[d])+" ",r=o.text(l,-9999,-9999).css(t).add(),d++,n.match(/<\s*\/?\s*br\/?\s*>/)){words.splice(d-1,1);break}p=r.getBBox().width}while(p<=i&&d<words.length);d-1==c?(s=r.element.offsetHeight,a=words[d-1],c=d):(a=words.slice(c,d-1).join(" "),O||(a=a.replace(/\s+</g,"<").replace(/>\s+/g,">")),c=d-1),h.push({line:a,lineHeight:s}),r.destroy(),r=null}return h},generateCSV:function(){function t(e){if(!O.hasOwnProperty(e)){if(k){for(o=S.length-1;o>=0&&!(e<S[o]);o--);S.splice(o+1,0,e)}else S.push(e);O[e]=1}}var i,o,r,s,n,a,l,p,h,c,d,u,g,y,x,m=this,f=m.options.exporting.csvOptions.overrideURL,b=/,/g,v=/\n/g,T=/<[\/\w\s]+>/g,A=C(m.options.xAxis)[0],w=!0,k=!1,S=[],_={},L={},O={},M=!1,P=!1,D=this.cleanTitleForCSVExport();if(f)return window.location=f;try{k="datetime"==A.type}catch(e){}try{void 0!==A.title.text&&A.title.text.length>0&&(a=A.title.text)}catch(e){}for(F=0;F<m.series.length;F++)if(n=m.series[F],m.options.navigator){if(!m.options.navigator.enabled||"Navigator"!=n.name){if(!1===n.visible)continue;if(null==n.linkedParent||!m.options.exporting.combineLinkedSeriesInExport)if(u=_[I=n.name+(n.yAxis&&n.yAxis.axisTitle?" "+n.yAxis.axisTitle.textStr:"")]={},"pie"==n.type)for(i=0;i<n.points.length;i++)u[r=(s=n.points[i]).name]=e.numberFormat(s.y,n.options.tooltip.precision?n.options.tooltip.precision:void 0,".",""),t(r);else{$=n.points;if(m.options.exporting.combineLinkedSeriesInExport)for(N=0;N<n.linkedSeries.length;N++)$=$.concat(n.linkedSeries[N].points);else if(null!=n.linkedParent){for(var E=[],F=0;F<n.linkedParent.points.length;F++)n.linkedParent.points[F].y&&E.push(n.linkedParent.points[F].y);E[E.length-1]===$[0].y&&$.splice(0,1)}for(i=0;i<$.length;i++)"string"==typeof(r=void 0!==(s=$[i]).category?s.category:s.x)&&(r='"'+r.replace(T," ")+'"'),u[r]=e.numberFormat(n.options.compare?s.change:s.y,n.options.tooltip.precision?n.options.tooltip.precision:void 0,".",""),t(r)}}}else if("range"==n.options.type||"arearange"==n.options.type){var H=n.name+" Low "+(n.yAxis&&n.yAxis.axisTitle?" "+n.yAxis.axisTitle.textStr:""),U=n.name+" High "+(n.yAxis&&n.yAxis.axisTitle?" "+n.yAxis.axisTitle.textStr:"");for(dataHash1=_[H]={},dataHash2=_[U]={},L[H]={sourceKey:n.options.sourceKey,sourceLink:n.options.sourceLink},L[U]={sourceKey:n.options.sourceKey,sourceLink:n.options.sourceLink},M|=void 0!==L[H].sourceLink,P|=void 0!==L[H].sourceKey,i=0;i<n.data.length;i++)if(!1!==n.data[i].export&&(r=void 0!==(s=n.data[i]).category?s.category:s.name,dataHash1[r]=e.numberFormat(n.options.compare?s.change:s.low,n.options.tooltip.precision?n.options.tooltip.precision:void 0,".",""),dataHash2[r]=e.numberFormat(n.options.compare?s.change:s.high,n.options.tooltip.precision?n.options.tooltip.precision:void 0,".",""),!O.hasOwnProperty(r))){if(k){for(o=S.length-1;o>=0&&!(r<S[o]);o--);S.splice(o+1,0,r)}else S.push(r);O[r]=1}}else if(null==n.linkedParent){var I;u=_[I=n.name+(n.yAxis&&n.yAxis.axisTitle?" "+n.yAxis.axisTitle.textStr:"")]={},L[I]={sourceKey:n.options.sourceKey,sourceLink:n.options.sourceLink},M|=void 0!==L[I].sourceLink,P|=void 0!==L[I].sourceKey;for(var $=n.data,N=0;N<n.linkedSeries.length;N++)$=$.concat(n.linkedSeries[N].data);for(i=0;i<$.length;i++)if(!1!==$[i].export&&(u[r=void 0!==(s=$[i]).category?s.category:s.name]=e.numberFormat(n.options.compare?s.change:s.y,n.options.tooltip.precision?n.options.tooltip.precision:void 0,".",""),!O.hasOwnProperty(r))){if(k){for(o=S.length-1;o>=0&&!(r<S[o]);o--);S.splice(o+1,0,r)}else S.push(r);O[r]=1}}for(y=m.options.maxXType,void 0===a&&(a=e.csvXLabels[y]||e.csvXLabels[e.xTypes.category]),h=(h=[[D]]).concat(this.generateCSVHeaders()),M&&h.push(p=["Series Link"]),P&&h.push(x=["Series Key"]),h.push(c=[a]),g=h.length,F=0;F<S.length;F++){for(o in i=S[F],d=h[F+g]=[e.parseXValue(y,i,m.userOptions.timezone)],_)_.hasOwnProperty(o)&&(u=_[o],l=L[o],w&&(c.push(o.replace(b,"").replace(v," ").replace(T," ")),M&&p.push(void 0!==l.sourceLink?l.sourceLink:""),P&&x.push(void 0!==l.sourceKey?l.sourceKey:"")),u.hasOwnProperty(i)?null==u[i]?d.push("--"):d.push(u[i]):d.push(""));w=!1}this.postCSVData(h,D.replace(/\s+/g,"_"))},cleanTitleForCSVExport:function(){var e=this,t="chartData",i=/[\?%\*:\|"\<\>\\,]/g,o=/\//g,r=/\s+/g,s=/<[\/\w]+>/gi;try{void 0!==e.options.title.text&&e.options.title.text.length>0&&(t=e.options.title.text.replace(s," ").replace(i,"").replace(o," per ").replace(r," ")),void 0!==e.options.subtitle.text&&e.options.subtitle.text.length>0&&(t+=" ("+e.options.subtitle.text.replace(s," ").replace(i,"").replace(o," per ").replace(r," ")+")")}catch(e){}return t},generateCSVHeaders:function(){var e=[];return e.push([window.location.href.toString()]),e.push([(new Date).toTimeString()]),e.push([this.options.credits.text]),e},postCSVData:function(e,t){var i,o,r=this.options.exporting.csvOptions.postURL;i=$('<input type="hidden" name="csvArr"/>').val(JSON.stringify(e)),o=$('<input type="hidden" name="fileName"/>').val(t),jQuery('<form style="display:none;" action="'+r+'" method="post"></form>').append(i).append(o).appendTo("body").submit().remove()},addButton:function(e){var t,i,o=this,r=o.renderer,s=w(o.options.navigation.buttonOptions,e),a=s.onclick,l=s.menuItems,h={stroke:s.symbolStroke,fill:s.symbolFill},c=s.symbolSize||12;if(o.btnCount||(o.btnCount=0),o.btnCount++,o.exportDivElements||(o.exportDivElements=[],o.exportSVGElements=[],o.exportEmbedElements=[]),!1!==s.enabled){o.buttonGroup||(o.buttonGroup=r.g("buttonGroup").add(),o.exportSVGElements.push(o.buttonGroup));var d,u=s.theme,g=u.states,y=g&&g.hover,x=g&&g.select;delete u.states,a?d=function(){a.apply(o,arguments);for(var e=0;e<o.exportDivElements.length;e++)$(o.exportDivElements[e]).on("mousedown",(function(e){return e.stopPropagation(),!1}))}:l&&(d=function(){o.contextMenu("contextmenu",l,i.translateX,i.translateY,i.width,i.height,i);for(var e=0;e<o.exportDivElements.length;e++)$(o.exportDivElements[e]).on("mousedown",(function(e){return e.stopPropagation(),!1}));i.setState(2)}),s.text&&s.symbol?u.paddingLeft=Highcharts.pick(u.paddingLeft,25):s.text||p(u,{width:s.width,height:s.height,padding:0}),i=r.button(s.text,0,0,d,u,y,x).attr({title:o.options.lang[s._titleKey],"stroke-linecap":"round"}),s.symbol&&(t=r.symbol(s.symbol,s.symbolX-c/2,s.symbolY-c/2,c,c).attr(p(h,{"stroke-width":s.symbolStrokeWidth||1,zIndex:1})).add(i)),i.add().align(p(s,{width:i.width,x:Highcharts.pick(s.x,n)}),!0,"spacingBox"),n+=(i.width+s.buttonSpacing)*("right"===s.align?-1:1),o.exportSVGElements.push(i,t)}}}),p(u.prototype,{getOffset:function(){var e,i,o,r,s,n,a,l,p,c,d,u,y,m=this,b=m.chart,C=b.renderer,w=m.options,k=m.tickPositions,S=m.ticks,_=m.horiz,L=m.side,O=b.inverted?[1,0,3,2][L]:L,M=0,P=0,D=w.title,E=w.labels,F=0,H=b.axisOffset,U=b.clipOffset,I=[-1,1,1,-1][L],$=1,N=g(E.maxStaggerLines,5);if(m.hasData=e=m.hasVisibleSeries||A(m.min)&&A(m.max)&&!!k,m.showAxis=i=e||g(w.showEmpty,!0),m.staggerLines=m.horiz&&E.staggerLines,m.axisGroup||(m.gridGroup=C.g("grid").attr({zIndex:w.gridZIndex||1}).add(),m.axisGroup=C.g("axis").attr({zIndex:w.zIndex||2}).add(),m.labelGroup=C.g("axis-labels").attr({zIndex:E.zIndex||7}).addClass(x+m.coll.toLowerCase()+"-labels").add()),e||m.isLinked){if(m.labelAlign=g(E.align||m.autoLabelAlign(E.rotation)),h(k,(function(e){S[e]?S[e].addLabel():S[e]=new f(m,e)})),m.horiz&&!m.staggerLines&&N&&!E.rotation){for(n=m.reversed?[].concat(k).reverse():k;$<N;){for(a=[],l=!1,s=0;s<n.length;s++)p=n[s],y=s%$,(u=(c=S[p].label&&S[p].label.getBBox())?c.width:0)&&(d=m.translate(p),a[y]!==t&&d<a[y]&&(l=!0),a[y]=d+u);if(!l)break;$++}$>1&&(m.staggerLines=$)}h(k,(function(e){0!==L&&2!==L&&{1:"left",3:"right"}[L]!==m.labelAlign||(F=T(S[e].getLabelSize(),F))})),m.staggerLines&&(F*=m.staggerLines,m.labelOffset=F)}else for(r in S)S[r].destroy(),delete S[r];D&&D.text&&!1!==D.enabled&&(m.axisTitle||(m.axisTitle=C.text(D.text,0,0,D.useHTML).attr({zIndex:7,rotation:D.rotation||0,align:D.textAlign||{low:"left",middle:"center",high:"right"}[D.align]}).addClass(x+this.coll.toLowerCase()+"-title").css(D.style).add(m.axisGroup),m.axisTitle.isNew=!0),i&&(M=m.axisTitle.getBBox()[_?"height":"width"],P=g(D.margin,_?5:10),o=D.offset),m.axisTitle[i?"show":"hide"]()),m.offset=I*g(w.offset,H[L]),m.axisTitleMargin=g(isNaN(o)?o:o+F+P,F+P+(2!==L&&F&&I*w.labels[_?"y":"x"])),H[L]=T(H[L],m.axisTitleMargin+M+I*m.offset),U[O]=T(U[O],2*v(w.lineWidth/2))}}),o=e.eiaTheme.yAxis,r=e.eiaTheme.xAxis,e.setOptions(e.eiaTheme);function K(){var t,i={eia_logo1:e.eia_yellow,eia_logo2:e.eia_green,eia_logo3:e.eia_blue,eia_logo4:e.eia_dk_grey,eia_logo5:e.eia_dk_grey,eia_logo6:"white"},o={eia_logo3:function(e,t,i){return["M",e+41.61*i,t+8.68*i,"C",e+34.49*i,t+4.67*i,e+23.03*i,t+5.18*i,e+1.84*i,t+17.02*i,"C",e+.68*i,t+17.67*i,e+.67*i,t+17.31*i,e+.95*i,t+16.97*i,"C",e+2.42*i,t+15.22*i,e+21.34*i,t+-.32*i,e+38.13*i,t+1.15*i,"C",e+43.03*i,t+1.58*i,e+51.75*i,t+4.42*i,e+53.2*i,t+13.88*i,"C",e+54.65*i,t+23.32*i,e+50.36*i,t+28.88*i,e+49.07*i,t+29.18*i,"C",e+50.78*i,t+22.6*i,e+49.67*i,t+13.22*i,e+41.61*i,t+8.68*i,"Z"]},eia_logo4:function(e,t,i){return["M",e+20.43*i,t+17.17*i,"C",e+20.43*i,t+19.1946*i,e+22.6217*i,t+20.4599*i,e+24.375*i,t+19.4476*i,"C",e+25.1887*i,t+18.9778*i,e+25.69*i,t+18.1096*i,e+25.69*i,t+17.17*i,"C",e+25.69*i,t+15.1454*i,e+23.4983*i,t+13.8801*i,e+21.745*i,t+14.8924*i,"C",e+20.9313*i,t+15.3622*i,e+20.43*i,t+16.2304*i,e+20.43*i,t+17.17*i]},eia_logo5:function(e,t,i){return["M",e+17.69*i,t+29.88*i,"C",e+17.75*i,t+24.43*i,e+14.74*i,t+20.56*i,e+9.3*i,t+20.58*i,"C",e+3.97*i,t+20.52*i,e+.73*i,t+24.64*i,e+.81*i,t+29.89*i,"C",e+.81*i,t+32.7*i,e+1.61*i,t+34.96*i,e+3.2*i,t+36.66*i,"C",e+5.2*i,t+38.96*i,e+9.01*i,t+39.67*i,e+12.07*i,t+38.9*i,"C",e+14.28*i,t+38.37*i,e+16.01*i,t+36.87*i,e+17.27*i,t+34.96*i,"L",e+13.8*i,t+33.05*i,"C",e+11.22*i,t+37.35*i,e+5.28*i,t+35.86*i,e+5.05*i,t+30.83*i,"H",e+17.67*i,"L",e+17.69*i,t+29.87*i,"Z","M",e+5.32*i,t+27.51*i,"C",e+6.28*i,t+23.27*i,e+12.61*i,t+23.07*i,e+13.5*i,t+27.51*i,"H",e+5.31*i,"Z","M",e+21.02*i,t+22.35*i,"V",t+39.04*i,"H",e+25.1*i,"V",t+22.35*i,"H",e+21.02*i,"Z","M",e+44.23*i,t+28.13*i,"C",e+44.25*i,t+21.79*i,e+40.9*i,t+20.58*i,e+38.04*i,t+20.37*i,"C",e+33.32*i,t+20.02*i,e+29.84*i,t+22.91*i,e+29.84*i,t+22.91*i,"L",e+32.43*i,t+25.56*i,"C",e+35.19*i,t+23.35*i,e+39.85*i,t+23.81*i,e+40.22*i,t+26.69*i,"C",e+40.22*i,t+26.69*i,e+36.14*i,t+27.61*i,e+35.24*i,t+27.77*i,"C",e+32.4*i,t+28.37*i,e+30.54*i,t+28.62*i,e+29.05*i,t+30.72*i,"C",e+27.76*i,t+32.75*i,e+28.1*i,t+35.86*i,e+30*i,t+37.54*i,"C",e+32.63*i,t+39.87*i,e+38.07*i,t+39.87*i,e+40.8*i,t+36.72*i,"C",e+40.8*i,t+36.72*i,e+41.14*i,t+38.49*i,e+41.38*i,t+39.04*i,"H",e+45.49*i,"C",e+43.76*i,t+35.07*i,e+44.22*i,t+30.81*i,e+44.23*i,t+28.13*i,"Z","M",e+40.22*i,t+30.98*i,"C",e+39.99*i,t+32.84*i,e+38.74*i,t+35.09*i,e+35.33*i,t+35.53*i,"C",e+33.23*i,t+35.8*i,e+31.23*i,t+34.43*i,e+32.28*i,t+32.5*i,"C",e+32.61*i,t+31.9*i,e+33.36*i,t+31.43*i,e+34.34*i,t+31.29*i,"C",e+34.94*i,t+31.2*i,e+35.61*i,t+31.13*i,e+36.31*i,t+31.01*i,"C",e+37.85*i,t+30.76*i,e+39.34*i,t+30.47*i,e+40.22*i,t+30.13*i,"C",e+40.22*i,t+30.13*i,e+40.25*i,t+30.76*i,e+40.22*i,t+30.98*i,"Z"]},eia_logo6:function(e,t,i){return["M",e+5.32*i,t+27.51*i,"C",e+6.28*i,t+23.27*i,e+12.61*i,t+23.07*i,e+13.5*i,t+27.51*i,"H",e+5.31*i,"Z","M",e+40.22*i,t+30.98*i,"C",e+39.99*i,t+32.84*i,e+38.74*i,t+35.09*i,e+35.33*i,t+35.53*i,"C",e+33.23*i,t+35.8*i,e+31.23*i,t+34.43*i,e+32.28*i,t+32.5*i,"C",e+32.61*i,t+31.9*i,e+33.36*i,t+31.43*i,e+34.34*i,t+31.29*i,"C",e+34.94*i,t+31.2*i,e+35.61*i,t+31.13*i,e+36.31*i,t+31.01*i,"C",e+37.85*i,t+30.76*i,e+39.34*i,t+30.47*i,e+40.22*i,t+30.13*i,"C",e+40.22*i,t+30.13*i,e+40.25*i,t+30.76*i,e+40.22*i,t+30.98*i,"Z"]}};for(t in o)o.hasOwnProperty(t)&&(symbolFn=o[t],this.renderer.symbols[t]=symbolFn,this.renderer.symbol(t,0,0,.6).attr({stroke:"none",fill:i[t]}).add().align({verticalAlign:"bottom",align:"left",y:-35,x:12}))}function q(){this.renderer.image("https://www.eia.gov/global/images/logos/Thomson_Reuters_logo.png",0,0,135,30).align({verticalAlign:"bottom",align:"left",y:-35,x:5}).add()}function J(){return!1}function ee(t,i){var o="",r=new Date(i);switch(t){case e.xTypes.annual:o=r.getUTCFullYear().toString();break;case e.xTypes.monthly:o=r.getUTCFullYear().toString()+(r.getUTCMonth()+1<10?"0":"")+(r.getUTCMonth()+1).toString();break;case e.xTypes.weekly:case e.xTypes.daily:o=r.getUTCFullYear().toString()+(r.getUTCMonth()+1<10?"0":"")+(r.getUTCMonth()+1).toString()+(r.getUTCDate()<10?"0":"")+r.getUTCDate().toString();break;case e.xTypes.quarterly:o=r.getUTCFullYear().toString()+"Q"+Math.ceil((r.getUTCMonth()+1)/3).toString();break;case e.xTypes.hourly:o=r.getUTCFullYear().toString()+(r.getUTCMonth()+1<10?"0":"")+(r.getUTCMonth()+1).toString()+(r.getUTCDate()<10?"0":"")+r.getUTCDate().toString()+"T"+(r.getUTCHours()<10?"0":"")+r.getUTCHours()+"Z"}return o}bloomberg_logoURL="/global/images/logos/Bloomberg_logo.png",bloomberg_logoHeight=30,bloomberg_logoWidth=135,bloomberg_logoPadding=5,e.Renderer.prototype.symbols.download=function(e,t,i,o){var r=15,s=13;return["M",(e+12.006*i/r).toString()+","+(t+12.598*o/s).toString(),"L",(e+19.004*i/r).toString()+","+(t+12.598*o/s).toString(),"L",(e+19.004*i/r).toString()+","+(t+15*o/s).toString(),"L",(e+4.004*i/r).toString()+","+(t+15*o/s).toString(),"L",(e+4.004*i/r).toString()+","+(t+12.598*o/s).toString(),"L",(e+11.003*i/r).toString()+","+(t+12.598*o/s).toString(),"L",(e+11.002*i/r).toString()+","+(t+12.597*o/s).toString(),"L",(e+7.694*i/r).toString()+","+(t+7.597*o/s).toString(),"L",(e+10.575*i/r).toString()+","+(t+7.597*o/s).toString(),"L",(e+10.575*i/r).toString()+","+(t+2.597*o/s).toString(),"L",(e+12.574*i/r).toString()+","+(t+2.597*o/s).toString(),"L",(e+12.574*i/r).toString()+","+(t+7.597*o/s).toString(),"L",(e+15.316*i/r).toString()+","+(t+7.597*o/s).toString(),"Z"]},l.prototype.callbacks.push((function(){var t=this;switch(t.options.chart.logo){case e.logos.eia:K.call(t);break;case e.logos.reuters:q.call(t);break;case e.logos.bloomberg:J.call(t);case e.logos.none:}})),e.Chart.prototype.openAPIEmbed=function(){window.open("/opendata/v1/embed.php?"+this.getEmbedParams())},e.Chart.prototype.getEmbedOptions=function(){var e=this.options;return void 0!==e.exporting&&void 0!==e.exporting.embedOptions?e.exporting.embedOptions:null},e.Chart.prototype.setEmbedOptions=function(e){var t=this.options;void 0!==t.exporting?void 0!==t.exporting.embedOptions?t.exporting.embedOptions=$.extend(!0,t.exporting.embedOptions,e):t.exporting.embedOptions=e:t.exporting={embedOptions:e}},e.Chart.prototype.getEmbedParams=function(){var e=this,t=this.options.exporting.embedOptions,i=["type=chart"];for(var o in t)t.hasOwnProperty(o)&&"start"!=o&&"end"!=o&&"date_mode"!=o&&i.push(o+"="+t[o]);if("datetime"==this.xAxis[0].options.type){var r=this.xAxis[0].dataMin,s=this.xAxis[0].dataMax,n=this.xAxis[0].min,a=this.xAxis[0].max,l=Math.max(e.xAxis[0].min,e.xAxis[0].dataMin),p=Math.min(e.xAxis[0].max,e.xAxis[0].dataMax);r<n&&s<=a?(i.push("date_mode=periods"),i.push("start="+ee(e.options.minXType,l)),i.push("end="+ee(e.options.minXType,p))):r<n&&s>a?(i.push("date_mode=range"),i.push("start="+ee(e.options.minXType,l)),i.push("end="+ee(e.options.minXType,p))):t.hasOwnProperty("date_mode")?(i.push("date_mode="+t.date_mode),"range"!=t.date_mode&&"start"!=t.date_mode||(t.hasOwnProperty("start")?i.push("start="+t.start):i.push("start="+ee(e.options.minXType,l))),"range"==t.date_mode&&(t.hasOwnProperty("end")?i.push("end="+t.end):i.push("end="+ee(e.options.minXType,p))),"periods"==t.date_mode&&t.hasOwnProperty("periods")&&i.push("periods="+t.periods)):i.push("date_mode=all")}return i.join("&")},e.wrap(e.Chart.prototype,"init",(function(t,i,s){var l=i,p=this,h=i||{};if(a=null!=l.chart.type&&"pie"==l.chart.type,this.axisZerosAligned=0,p.seriesColorQueue=Highcharts.getOptions().colors.slice(0),n=0,ops=jQuery.extend(!0,{},e.getOptions(),l),void 0!==ops.exporting.embedOptions){for(var d=ops.exporting.buttons.contextButton.menuItems,u=!1,g={text:"Embed Chart",onclick:function(){this.openAPIEmbed()}},y=0;y<d.length;y++)if(null==d[y]){d[y]=g,u=!0;break}u||d.push(g)}return ops.xAxis=c(C(ops.xAxis||{}),(function(e){return w(r,e,h.xAxis||{})})),ops.yAxis=c(C(ops.yAxis||{}),(function(e){return opposite=e.opposite,w(o,e,h.yAxis||{})})),function(t){switch(t.chart.logo){case e.logos.eia:t.chart.forExport||(t.chart.spacingBottom+=35),-5==t.credits.position.y&&(t.credits.position.y=-35/3),t.credits.position.x=55;break;case e.logos.reuters:t.chart.forExport||(t.chart.spacingBottom+=40),t.credits.position.x=150,t.credits.position.y=-35/3;break;case e.logos.bloomberg:t.chart.forExport||(t.chart.spacingBottom+=bloomberg_logoHeight+2*bloomberg_logoPadding),t.credits.position.x=bloomberg_logoWidth+bloomberg_logoPadding+10,t.credits.position.y=-(bloomberg_logoHeight+bloomberg_logoPadding)/3;case e.logos.none:}}(ops),function(e){if(!e.chart.outerContainer&&!e.chart.printContainer){var t=e.chart.renderTo,i="string"==typeof t?$("#"+t):$(t),o=i.closest(".outerChartContainer"),r=i.closest(".printChartContainer");0==o.length&&0==r.length&&(o=$("<div></div>").css({height:"auto",width:"auto"}).addClass("outerChartContainer"),r=$("<div></div>").css({height:"auto",width:"auto"}).addClass("printChartContainer"),i.before(o),r.appendTo(o),i.appendTo(r)),e.chart.outerContainer=o[0],e.chart.printContainer=r[0],e.chart.container=i[0]}}(ops),i=a?M.call(p,t,ops,s):I.call(p,t,ops,s),t.call(this,i,s)})),e.wrap(e.Chart.prototype,"addSeries",(function(t,i,o,r){var s,n=this,a=i,l=new e.Series,p=n.options;a.color||(a.color=F.apply(n));var h=H.call(this,a);l.init(n.dummyChart,$.extend(!0,{},a,p.plotOptions.series)),s=R.call(n.dummyChart,l),a.xType=l.options.xType=void 0!==a.xType?a.xType:s.xType,a.yType=l.options.yType=void 0!==a.yType?a.yType:s.yType,D.call(n,p),s.xType<=4&&s.xType>=1&&p.chart.breakLines&&U(l);var c=t.call(this,i,o,r);if(h){h.linkedTo=c.index;var d=this.addSeries(h);c.linkedSeries.push(d)}return Z.call(n),c})),e.wrap(e.Series.prototype,"setData",(function(e,t,i,o,r){var s=this.chart,n=this.options,a=$.extend(!0,{},n,{data:t}),l=null;!a.projection&&s.dummyChart&&null!==(l=H.call(s,a))&&!0===l.projection&&this.linkedSeries&&this.linkedSeries.length>0&&e.call(this.linkedSeries[0],l?l.data:[],i,o,r);var p=e.call(this,a.data,i,o,r);return Z.call(s),p})),e.wrap(e.Series.prototype,"remove",(function(e,t,i){var o=this,r=o.chart,s=r.options;if(o.linkedSeries)for(var n=o.linkedSeries.length-1;n>=0;n--)o.linkedSeries[n].remove();r.dummyChart.series.splice(o.index,1);var a=e.call(this,t,i);D.call(r,s);var l=-1/0;for(n=0;n<r.series.length;n++)r.scroller&&r.series[n]===r.scroller.series||(l=Math.max(l,r.options.colors.indexOf(r.series[n].color)));return r.counters.color=l>=0?l:0,Z.call(r),a})),e.wrap(e.Series.prototype,"getColor",(function(e){return e.call(this)})),e.wrap(e.Chart.prototype,"setTitle",(function(t,i,o,r){var s=$("<div></div>").appendTo("body").css({position:"absolute",top:-9999}),n=new e.Renderer(s[0],0,0),a=this,l=a.options,p=l.chart.titleAdjust,h=a.container;return void 0!==(i=w({},l.title,i))&&null!=i&&(p=z.call(a,i,n,(l.chart.width||$(h).width()||600)-(!1!==l.exporting.enabled?44+(null==l.exporting.buttons.contextButton.text?0:100):0)-(l.chart.spacing[1]+l.chart.spacing[3])),l.chart.titleAdjust=p),void 0!==o&&null!=o&&(o.y=(isNaN(p)?0:p)+(isNaN(l.chart.spacingTop)?l.chart.spacing[0]:l.chart.spacingTop)+5),s.remove(),t.call(this,i,o,r)})),e.wrap(e.Axis.prototype,"adjustTickAmount",(function(e){var t=this,i=t.tickPositions?t.tickPositions.length:t.tickAmount,o=t._maxTicksKey,r=t.chart.maxTicks,s=e.call(this);if(r&&r[o]&&i!=t.tickPositions.length&&this.chart.axisZerosAligned&&this.chart.options.chart.lineUpYAxisZeros){t.tickPositions.length,t.tickInterval;for(var n,a=0;a<t.chart.axes.length;a++)if(t.chart.axes[a]!==t&&t.chart.axes[a].coll==t.coll&&t.chart.axes[a].tickPositions.length==r[o]){n=t.chart.axes[a];break}var l=$.inArray(0,n.tickPositions),p=$.inArray(0,t.tickPositions);if(l>=0&&p>=0&&l!=p){for(a=0;a<Math.abs(l-p);a++)t.tickPositions.unshift(t.tickPositions[0]-t.tickInterval),t.tickPositions.splice(t.tickPositions.length-1,1);t.min=t.tickPositions[0],t.max=t.tickPositions[t.tickPositions.length-1]}}return s})),e.wrap(e.Chart.prototype,"contextMenu",(function(e,t,i,o,r,s,n,a){this["cache-"+t];return e.call(this,t,i,o,r,s,n,a)})),e.wrap(e.Chart.prototype,"getSVG",(function(e,t){var i=e.call(this,t);return i=i.replace(/></g,">\r\n<")}))}(Highcharts);    if(/\/hist\//.test(window.location.pathname)){
		/**
 * The Petroleum Navigator leaf page charting function
**/
var leafHandler = function($){
	return function(){
		// Is this the Natural Gas Navigator?
		var isNG, isPET, sourcekey, frequency, frequency2, navigatorPrefix;
		
		//min must not be zero if negative values
		negativeValues = false;
		// An array of months for performing lookups
		var monthsArray = [
			'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
			'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
		];
		// Returns the Chart Title and Units as an array
		var getTitleMatch = function(){
			//var titleRegex = /[^\(\)]+/g;
			var title = $('.Title1').text();
			//return title.match(titleRegex);  <- this didn't work as some series name have parenthical phrases
			var aTitle = [title.substring(0,title.lastIndexOf('(')-1), title.substring(title.lastIndexOf('(')+1,title.lastIndexOf(')')) ];
			return aTitle;
		}
		// Returns the table rows in a page for iteration
		var getTableRows = function(){
			if(isNG) {
				return $('body > table:eq(2) tr');
			} else if(isPET) {
				if (frequency2=='m') {
					return $('table.FloatTitle tr');
				} else {
					return $('body > table:eq(2) tr');	//y
				}
			} else {
				return $('table.FloatTitle tr');
				
			}	
		}
		// Trims null values from the beginning and end of data
		// arrays
		var trimNulls = function(data){
			var i;
			var dataCopy = data.slice(0);
			
			for(i=data.length - 1; i>0; i--){
				if(data[i][1] === null) dataCopy.pop();
				else break;
			}
			for(i=0; i<data.length; i++){
				if(data[i][1] === null) dataCopy.shift();
				else break;
			}

			return dataCopy;
		}
		// Creates the options object for highcharts.  Any
		// Changes to the chart options should be implemented
		// Here
		var createOptions = function(parent, title, units, data){
			var ret;
		
			sourcekey = determineSourceKey().toUpperCase();
			frequency = determinePeriodType().toUpperCase();
			navigatorPrefix = isNG ? 'NG' : 'PET';
			
			var titleMatch = getTitleMatch();
			ret = {
				chart:{
					zoomType:'xy',
					renderTo:parent.id
				},
				title:{
					text:title
				},
				xAxis:{
					type:'datetime'
				},
				yAxis:{
					title:{
						text:units
					}
				},
				series:[{
					name: title,
					data: data
				}],
				legend:{
					enabled:true
				},
				exporting : {
					embedOptions : {
						series_id : navigatorPrefix + '.' + sourcekey + '.' + frequency
					}
				}/*,   This shows single points, but is too slow in IE for vary large series
				plotOptions:{
					line:{
						marker:{enabled: true, radius: 1}
					}
				}  */
			};
	
			switch(frequency){
				case 'd': // Daily
					ret.xAxis.maxZoom = 10 * 24 * 3600000;
					break;
				case 'w': // Weekly
				case '4': // 4-Week Avg
					ret.xAxis.maxZoom = 10 * 7 * 24 * 3600000;
					break;
				case 'm': // Monthly
					ret.xAxis.maxZoom = 10* 30 * 24 * 3600000;
					break;
				case 'a': // Annual
					ret.xAxis.maxZoom = 10 * 365 * 24 * 3600000;
					break;
			}
			if(negativeValues) ret.yAxis.min = null;
			
			if(isReuters()){
				ret.chart.logo =2;
				ret.credits = {text: 'Data Source: Thomson Reuters', href: '', style: {color: '#FFFFFF'}};
			}
			else if(isBloomberg()){
				ret.chart.logo =3;
				ret.credits = {text: 'Source: Bloomberg', href: '', style: {color: '#000000'}};
			}
			if(navigator.userAgent.toLowerCase().indexOf('chrome') > -1)console.info(JSON.stringify(ret));
			return ret; 
		}
		//Indicates whether this is a Reutures series
		var isReuters = function(){
		  var tr_series = ['RWTC','RBRTE','EER_EPMRU_PF4_Y35NY_DPG','EER_EPMRU_PF4_RGC_DPG','EER_EPMRR_PF4_Y05LA_DPG','EER_EPD2F_PF4_Y35NY_DPG','EER_EPJK_PF4_RGC_DPG','EER_EPLLPA_PF4_Y44MB_DPG', 'RNGWHHD'];

		  for(var j=0;j<tr_series.length;j++){
			  if(sourcekey == tr_series[j]) return true;
		  }
		  
		  return false;
		}
		//Indicates whether this is a Reutures series
		var isBloomberg = function(){
		  var tr_series = ['NGM_EPG0_PLC_NUS_DMMBTU'];

		  for(var j=0;j<tr_series.length;j++){
			  if(sourcekey == tr_series[j]) return true;
		  }
		  
		  return false;
		}
		
		// Parses string values into floats
		var parseValue = function(string){
			var val, ret;
			val = parseFloat(string.replace(/,/g, ''));
			if(isNaN(val)){
				if(string == '')
					ret = void 0;
				else
					ret = null;
			}
			else ret = val;
			
			return ret;
		};
		// The processing function for processing daily data
		var processDailyData = function(){
			var data = [];
			var dateRegex = /(\d{4})\s+(\w{3})-(\s?\d{1,2})/ // (yyyy) (mmm)-(dd)
			
			getTableRows().each(function(i, n){
				var dateString = $(n).find('td.B6').html();
				if(dateString != void 0){
					var dateMatch = dateString.match(dateRegex);
					var currentDate = Date.UTC(
						dateMatch[1], // Year
						jQuery.inArray(dateMatch[2], monthsArray), // Month 
						dateMatch[3] // Day
					);
					$(n).find('td.B3').each(function(j, o){
						var val = parseValue($(o).html());
						if(val < 0) negativeValues = true;
						if(val !== void 0) data.push([currentDate, val]);
						currentDate = currentDate + 864E5; // Add a day
					});
				}
			});
			
			return data;
		}
		// The processing function for weekly data
		var processWeeklyData = function(){
			var data = [];
			var dateRegex1 = /(\d{4})-(\w{3})/; // yyyy-mmm
			var dateRegex2 = /\d{2}/g; // mm and dd
			
			getTableRows().each(function(i, n){
				var dateString1 = $(n).find('td.B6').html();
				if(dateString1 != void 0){
					var dateMatch1 = dateString1.match(dateRegex1);
					$(n).find('td.B5').each(function(j, o){
						var dateString2 = $(o).html();
						var dateMatch2 = dateString2.match(dateRegex2);
						if(dateMatch2 !== null){
							var currentDate = Date.UTC(
								dateMatch1[1], // Year
								jQuery.inArray(dateMatch1[2], monthsArray), // Month
								dateMatch2[1] // Day
							);
							var val = parseValue($(o).next('td.B3').html());
							if(val < 0) negativeValues = true;
							if(val !== void 0) data.push([currentDate, val]);
						}
					});
				}
			});
			return data;
		}
		// The processing function for monthly data
		var processMonthlyData = function(){
			var data = [];
			var dateRegex = /(\d{4})/; // yyyy
			
			getTableRows().each(function(i, n){
				var dateString = $(n).find('td.B4').html();
				if(dateString != void 0){
					var dateMatch = dateString.match(dateRegex);
					$(n).find('td.B3').each(function(j, o){
						var currentDate = Date.UTC(
							dateMatch[0], // Year
							j, // Column Index = Month Number
							1 // 1st
						);
						var val = parseValue($(o).html());
						if(val < 0) negativeValues = true;
						if(val !== void 0) data.push([currentDate, val]);
					});
				}
			});
			
			return data;
		}
		// The processing function for annual data
		var processAnnualData = function(){
			var data = [];
			var dateRegex = /(\d{4})/; // yyyy
			
			getTableRows().each(function(i, n){
				var dateString = $(n).find('td.B4').html();
				if(dateString != void 0){
					var dateMatch = dateString.match(dateRegex);
					var currentYear = dateMatch[0];
					$(n).find('td.B3').each(function(j, o){
						var currentDate = Date.UTC(
							currentYear, // Year
							0, // January
							1 // 1st
						);
						var val = parseValue($(o).html());
						if(val < 0) negativeValues = true;
						if(val !== void 0) data.push([currentDate, val]);
						currentYear++;
					});
				}
			});
			return data;
		}
		// Determines the period type
		function determinePeriodType () {
		// var determinePeriodType = function(){
			// console.log('isNG', isNG, window.location.search.match(/f=([\w\d]{1})/));
			var periodType = null;

			if (window.location.pathname.match(/([\w]{1})\.htm/)) {
				periodType = window.location.pathname.match(/([\w]{1})\.htm/)[1];
			}

			if (window.location.pathname.match(/([\w]{1})\.ashx/)) {
				// periodType = window.location.pathname.match(/([\w]{1})\.ashx/)[1];
				periodType = window.location.search.match(/f=([^&]+)/)[1];
			} 
			
			// if(true || isNG || isPET)
			// 	if (window.location.pathname.match(/([\w]{1})\.htm/)) {
			// 		periodType = window.location.pathname.match(/([\w]{1})\.htm/)[1];
			// 	}

			// 	if (window.location.pathname.match(/([\w]{1})\.ashx/)) {
			// 		// periodType = window.location.pathname.match(/([\w]{1})\.ashx/)[1];
			// 		periodType = window.location.search.match(/f=([^&]+)/)[1];
			// 	}
				
			// 	//  periodType = window.location.pathname.match(/([\w]{1})\.htm/)[1];
			// else
			// 	if (window.location.search.match(/f=([\w\d]{1})/) != null) {
			// 		periodType = window.location.search.match(/f=([\w\d]{1})/)[1];	
			// 	}
				// var periodType = window.location.search.match(/f=([\w\d]{1})/)[1];
			if (periodType==null) {
				return periodType;
			} else {
				return periodType.toLowerCase();
			}
			// return periodType.toLowerCase();
		}

		function getParameterByName(name, url) {
			if (!url) url = window.location.href;
			name = name.replace(/[\[\]]/g, "\\$&");
			var regex = new RegExp("[?&]" + name + "(=([^&#]*)|&|#|$)"),
				results = regex.exec(url);
			if (!results) return null;
			if (!results[2]) return '';
			return decodeURIComponent(results[2].replace(/\+/g, " "));
		}
		
		function determineSourceKey() { 
		// var determineSourceKey = function() {
			if (window.location.pathname.match(/([\w]{1})\.htm/)) {
				sourcekey = window.location.pathname.match(/([\w-_]+)\w\.htm/)[1];	
			} else if(window.location.pathname.match(/([\w]{1})\.ashx/)) {
				sourcekey = getParameterByName('s');
			}
			
			// if( true || isNG || isPET) {
			// 	// sourcekey = window.location.pathname.match(/([\w-_]+)\w\.htm/)[1];
			// 	if (window.location.pathname.match(/([\w]{1})\.htm/)) {
			// 		sourcekey = window.location.pathname.match(/([\w-_]+)\w\.htm/)[1];	
			// 	} else if(window.location.pathname.match(/([\w]{1})\.ashx/)) {
			// 		sourcekey = getParameterByName('s');
			// 	}

			// 	// sourcekey = window.location.pathname.match(/([\w-_]+)\w\.htm/)[1];
			// } else {
			// 	sourcekey = window.location.search.match(/s=([^&]+)/)[1];
			// }
			
			return sourcekey;
		}
		// Determines the appropriate processing function based on
		// the frequency parameter in the URL search parameters
		// var processData = function(periodType){
		var processData = function(){
			var processor;
			var periodType = determinePeriodType();
			// console.log(periodType);
			
	
			switch(periodType){
				case 'd': // Daily
					processor = processDailyData;
					// frequency ='d';
					break;
				case 'w': // Weekly
				case '4': // 4-Week Avg
					processor = processWeeklyData;
					break;
				case 'm': // Monthly
					processor = processMonthlyData;
					frequency2 = 'm';
					break;
				case 'a': // Annual
					processor = processAnnualData;
					break;
			}
			
			if (periodType != null) {
				return trimNulls(processor.apply());
			}
			// return trimNulls(processor.apply());
		};
		var prepareChartParent = function(){
			var imageParent, chartTable;
			if($('img[src*="hist_chart"]').length != 0) {				
				imageParent = $('img[src*="hist_chart"]').parent().html('');
				
				chartTable = imageParent.closest('table'),
				chartParent = $('<div id="highcharts_div" class="navigator-chart"></div>').width('100%').appendTo(imageParent);
			}
			else {
				imageParent = $('td[bgcolor="#f6dfb4"]').html('');
				$('td[bgcolor="#f6dfb4"]').removeAttr('height', 'bgcolor');
				chartTable = imageParent.closest('table'),
				chartParent = $('<div id="highcharts_div" class="navigator-chart" style="display:none;"></div>').width('100%').appendTo(imageParent);
			}
			// Make the chart 100% width
			chartTable.width('100%');
			// Hide the chart during processing
			chartParent.children().hide();
			return chartParent;
		}
		//This copies the referring pages div at the bottom of the page to the top of the page
		var copyReferringPages = function() {
			$('<div id="refer"></div>').html($('#refer').html()).insertAfter($('#contsub'));
			// remove first <BR> in body
			
			$('body > br').first().remove();
		}();
		
		// Add the source key and link to the api for this series
		var addSourceKeyIcon = function(chartParent) {
			var navigatorPrefix = isNG ? 'NG' : 'PET';
			
		$('<tr><td class="Footnotes" style="padding-top: 1em;"><a href="/opendata/series.php?sdid=' + navigatorPrefix + '.' + determineSourceKey().toUpperCase() + '.' + determinePeriodType().toUpperCase() + '"><span class="ico_sourcekey" style="margin-bottom:-3px;"></span></a>This series is available through the EIA open data API and can be downloaded to Excel or embedded as an interactive chart or map on your website.</td></tr>')
			.insertAfter(chartParent.closest('tr'));
		}
		
		// This is the function that creates the chart object.
		// Executes immediately i.e.: function(args){/*do*/}(args);
		var createLeafChart = function(){
			isNG = window.location.pathname.match(/\/ng\//) !== null;
			isPET = window.location.pathname.match(/\/pet\//) !== null;
			
			var chartParent = prepareChartParent();
			var titleMatch = getTitleMatch();
			var data = processData();
			addSourceKeyIcon(chartParent);
			leafChart = new Highcharts.Chart(
				createOptions(
					chartParent.get(0),
					titleMatch[0],
					titleMatch[1],
					data
				)
			);
		}();
	}
}(jQuery);/**
* 	Created by Shivan Computers Corporation on behalf of the
* 	U.S. Energy Information Administration:
*	Author : Ryan Lynch (Ryan.Lynch@eia.gov)
*	
*	EIA Analyis Module:
*
*	This module extend the EIA Theme to add various analytical
*	functions to the chart, along with a set of controls to 
*	activate them.
*
*	Depends on: highcharts.js, eia-theme.js
**/

(function(){
	
	// Shortcuts
	var HC = Highcharts,
	Chart = HC.Chart;
	each = HC.each,
	extend = HC.extend,
	map = HC.map,
	eiaTheme = HC.eiaTheme,
	seasonalPalletAlpha = 0.3;
	
	// Default Analysis Options
	HC.setOptions(extend(eiaTheme, {
		analysis: {
			enableMovingAverage:true,
			enableSeasonalAnalysis:true
		}
	}));
	
	function seasonalPalletConverter(h){
		return 'rgba(' + 
			parseInt(h.substring(1,3), 16) + ',' +
			parseInt(h.substring(3,5), 16) + ',' + 
			parseInt(h.substring(5), 16) + ',' +
			seasonalPalletAlpha + ')';
	}
	
	// Colors for seasonal analysis
	extend(HC, {
		eia_seasonal_pallet : map(eiaTheme.colors.slice(1), seasonalPalletConverter)
	});
	
	//////////////////////////
	// Analytical Functions //
	//////////////////////////
	
	extend(Chart.prototype,{
	
		addMovingAverage : function(index, duration){
			var chart = this, baseSeries = chart.series[index],
			interval = baseSeries.options.xType, baseXData = baseSeries.xData,
			baseYData = baseSeries.yData, avgData = [], helperArray = [],
			dayInterval, durationTitle, avg = total = i = j = calcDuration = 0;
			// If we are dealing with a series that has been destroyed, then
			// return
			if(!baseXData) return;
			// else
			switch(interval){
				case HC.xTypes.Annual:
				case HC.xTypes.Quarterly:
				case HC.xTypes.Monthly:
					return;
					break;
				case HC.xTypes.weekly:
					durationTitle = duration + ' Week';
					dayInterval = 7 * 24 * 60 * 60 * 1000;
					break;
				case HC.xTypes.daily:
					durationTitle = duration + ' Day';
					dayInterval = 1 * 24 * 60 * 60 * 1000;
					break;
			}

			for(i = duration; helperArray.length == 0; i++){
				for(j = i - duration; j < i; j++){
					if(
						baseYData[j] != null && 
						baseXData[j] >= baseXData[i] - dayInterval * duration
					){
						total += baseYData[j];
						helperArray.push({
							x : baseXData[j],
							y : baseYData[j]
						});
					}
				}
			}
			
			avgData.push([baseXData[i], total/helperArray.length]);
			
			for(i = duration + 1; i < baseXData.length; i++){
				if(
					helperArray[0].x <= baseXData[i] - dayInterval * duration
				){
					total -= helperArray[0].y;
					helperArray.shift();
				}
				
				j = i - 1
				
				if(
					baseYData[j] != null && 
					baseXData[j] >= baseXData[i] - dayInterval * duration
				){
					total += baseYData[j];
					helperArray.push({
						x : baseXData[j],
						y : baseYData[j]
					});
				}
				
				avgData.push([baseXData[i], total / helperArray.length]);
			}
			//chart.legend.enabled = true;
			return chart.addSeries({
				//name : baseSeries.name + durationTitle + ' Moving Average',
				name : durationTitle + ' Moving Average',
				data : avgData
			});
		},
		
		createSeasonalAnalysis : function(index, interval){
			var chart = this, baseSeries = chart.series[index], yearOffset,  
			baseXData = baseSeries.xData, baseYData = baseSeries.yData,
			i = j = 0, series = [], seriesData, xType = baseSeries.options.xType,
			serie, startDate, currentDate, firstDate, originalStartDate, lastDate,
			dateString;
			
			if(!baseXData) return;
			j = baseXData.length - 1;
			startDate = originalStartDate = new Date(baseXData[j]);
			for(i = 0; i <= interval && j > 0; i++){
				endDate = new Date(
					startDate.getUTCFullYear() - 1, 
					startDate.getUTCMonth(), 
					startDate.getUTCDate()
				)
				seriesData = [];
				lastDate = new Date(baseXData[j]);
				while(
					j >= 0 && 
					(!currentDate || currentDate.getTime() >= endDate.getTime())
				){
					currentDate = new Date(baseXData[j]);
					seriesData.push({
						x: Date.UTC(
							currentDate.getUTCFullYear() + i,
							currentDate.getUTCMonth(),
							currentDate.getUTCDate()),
						tooltipX : currentDate.getTime(),
						y: baseYData[j]
					});
					firstDate = currentDate;
					j--;
				}
				seriesData.sort(function(a,b) {
					if(a.x > b.x) {
						return 1;
					}
					else if(a.x < b.x) {
						return -1;
					}
					return 0;
					});
				j++;
				switch(xType){
					case HC.xTypes.daily:
						dateString = HC.dateFormat('%m/%d/%Y', firstDate.getTime()) + 
						' to ' + HC.dateFormat('%m/%d/%Y', lastDate.getTime());
						break;
					case HC.xTypes.weekly:
						dateString = HC.dateFormat('%m/%d/%Y', firstDate.getTime()) + 
						' to ' + HC.dateFormat('%m/%d/%Y', lastDate.getTime());
						break;
					case HC.xTypes.monthly:
						dateString = HC.dateFormat('%m/%Y', firstDate.getTime()) + ' to ' + 
						HC.dateFormat('%m/%Y', lastDate.getTime());
						break;
				}
				serie = {
					//name:baseSeries.options.name + ' ' + dateString,
					name: dateString,
					data:seriesData,
					legendIndex:i
				};
				if(i > 0) serie.color = HC.eia_seasonal_pallet[i-1];
				series.push(serie);
				startDate = endDate;
			}
			
			chart.removeAllSeries(false);
			
			for(j=series.length - 1; j>=0; j--){
				chart.addSeries(series[j], false);
			}
			
			chart.resetZoom();
			chart.redraw();
		}
	});
	
	function createSeasonalAnalysisOptions(s, index){
		var chart = this, xType = s.options.xType;
		if(s.type == 'line' && xType != HC.xTypes.category){
			var link, durations, maxInterval;
			if(xType > HC.xTypes.annual){
				maxInterval = (s.data[s.data.length - 1].x - s.data[0].x) / 31536E6 /*365 days*/;
				// Add moving averages
				each([5, 10], function(interval){
					if(interval < maxInterval){
						//option = $('<option></option>').html(s.name + ': ' + interval + ' Year Seasonal Analysis')
						option = $('<option></option>').html(interval + ' Year Seasonal Analysis')
						.data({
							method:chart.createSeasonalAnalysis,
							arguments:[index, interval]
						});
						option.appendTo(chart.options.chart.analysisSelect);
					}
				});
			}
		}
	}
	
	function createMovingAverageOptions(s, index){
		var chart = this, xType = s.options.xType;
		if(s.type == 'line' && xType != HC.xTypes.category){
			var link, durations,
			maxDuration = s.data.length;
			if(xType == HC.xTypes.daily){
				// Add moving averages
				each([30, 60, 90], function(duration){
					if(duration < maxDuration){
						//option = $('<option></option>').html(s.name + ': ' + duration + ' Day Moving Average ')
						option = $('<option></option>').html(duration + ' Day Moving Average ')
						.data({
							method:chart.addMovingAverage,
							arguments:[index, duration]
						});
						option.appendTo(chart.options.chart.analysisSelect);
					}
				});
			}
			if(xType == HC.xTypes.weekly){
				each([4, 8, 12], function(duration){
					if(duration < maxDuration){
						//option = $('<option></option>').html(s.name + ': ' + duration + ' Week Moving Average ')
						option = $('<option></option>').html(duration + ' Week Moving Average ')
						.data({
							method:chart.addMovingAverage,
							arguments:[index, duration]
						});
						option.appendTo(chart.options.chart.analysisSelect);
					}
				});
			}	
		}
	}
	
	function processAnalysis(){
		var chart = this, ops = chart.options,
		container = $(chart.options.chart.container),
		outerContainer = $(chart.options.chart.outerContainer);
		if (container.css("display") != 'none') {
			analysisContainer = $('<fieldset></fieldset>').css({
				height:'auto', 
				width:'auto', 
				// We set the margin and padding on the
				// captions container to match the chart
				// container so that the links are
				// lined up correctly
				marginLeft:container.css('marginLeft'),
				marginRight:container.css('marginRight')
				/*paddingLeft:container.css('paddingLeft'),
				paddingRight:container.css('paddingRight')*/
			}).addClass('chartAnalysisContainer').append('<legend>Chart Tools</legend>'),
			analysisSelect = $('<select><option>no analysis applied</option></select>'),
			analysisTable = $('<table></table>');
			
			// Enable or disable analysis based on maxXType
			switch(ops.maxXType){
				case HC.xTypes.annual:
					ops.analysis.enableMovingAverage &= false;
					ops.analysis.enableSeasonalAnalysis &= false;
					break;
				case HC.xTypes.monthly:
					ops.analysis.enableMovingAverage &= false;
					ops.analysis.enableSeasonalAnalysis &= true;
					break;
				case HC.xTypes.weekly:
					ops.analysis.enableMovingAverage &= true;
					ops.analysis.enableSeasonalAnalysis &= true;
					break;
				case HC.xTypes.daily:
					ops.analysis.enableMovingAverage &= true;
					ops.analysis.enableSeasonalAnalysis &= true;
					break;
				default:
					ops.analysis.enableMovingAverage &= false;
					ops.analysis.enableSeasonalAnalysis &= false;
					break;
			}
			
			if((
				ops.analysis.enableMovingAverage || 
				ops.analysis.enableSeasonalAnalysis
				) && !ops.chart.analysisContainer){
				analysisContainer.appendTo(outerContainer);
				analysisTable.append(
					$('<tr></tr>').append(
						$('<td></td>').append(analysisSelect)
					)
				).appendTo(analysisContainer);
				ops.chart.analysisContainer = analysisContainer[0];
				ops.chart.analysisSelect = analysisSelect[0];
				
				// Create Analytical function links
				resetButton = $('<button></button>').html('Reset').click(function(){
					chart.resetSeries(true);
				});
				applyButton = $('<button></button>').html('Apply').click(function(){
					var select = chart.options.chart.analysisSelect, 
					option = select.options[select.selectedIndex],
					optionData = $(option).data();
					chart.resetSeries(false);
					optionData.method.apply(chart, optionData.arguments);
				});
				if(
					ops.analysis.enableMovingAverage || 
					ops.analysis.enableSeasonalAnalysis
				){
					for(i in chart.series){
						s = chart.series[i];
						if(ops.analysis.enableMovingAverage) createMovingAverageOptions.call(chart, s, i);
						if(ops.analysis.enableSeasonalAnalysis) createSeasonalAnalysisOptions.call(chart, s, i);
					}
					if(analysisSelect.children().length > 0){
						analysisSelect.change(function(){
							if(analysisSelect.children(':selected').get(0).index ==0){
								chart.resetSeries(true);
							} else {
								var select = chart.options.chart.analysisSelect, 
								option = select.options[select.selectedIndex],
								optionData = $(option).data();
								chart.resetSeries(false);
								optionData.method.apply(chart, optionData.arguments);
							}
						});
						
						/*analysisTable.find('td')
							.append(applyButton)
							.append(resetButton);*/
						/*$('<tr></tr>').appendTo(analysisTable)
						.append(
							$('<td></td>')
							.append(applyButton)
							.append(resetButton)
						);*/
					}
					else
						analysisContainer.remove();
				}
			}
		}
		else {
			outerContainer.append('No chart available');
			outerContainer.css('height', '60px');
			outerContainer.css('background-color', '#f4f4f4');
			outerContainer.css('padding', '1em');
		}
	}
	
	Chart.prototype.callbacks.push(processAnalysis);
}());        leafHandlerLoaded = true;
        $(leafHandler);
    }
    else{
		var rootHandler = function($){
    var gbtn, cbtn, navigatorPrefix;
	
    function trimNulls(data){
        var i;
        var dataCopy = data.slice(0);
        
        for(i=data.length - 1; i>0; i--){
            if(data[i][1] === null) dataCopy.pop();
            else break;
        }
        for(i=0; i<data.length; i++){
            if(data[i][1] === null) dataCopy.shift();
            else break;
        }
        
        return dataCopy;
    }
    
    function addChartElements(){
        var cont = $('<div id="mce"></div')
            .css('display', 'none')
            .appendTo('body'),
        hccont = $('<div></div>')
            .attr('id', 'hc_container')
            .css({
                height : '600px',
                width : '800px',
                backgroundImage : 'url(/global/scripts/jQuery/tabs/loading.gif)',
                backgroundPosition : 'center',
                backgroundRepeat : 'no-repeat'
            })
            .appendTo(cont),
        link = $('<a></a>')
            .attr('href', '#hc_container')
            .attr('id', 'hc_link')
			.attr('title', 'Click and drag to zoom chart.  Use the buttons in upper right of chart for high quality print-outs and downloads.')
            .appendTo(cont)
            .fancybox({
				titlePosition : 'inside',
				titleFormat : function(title){
					return '<i>' + title + '</i>';
				},
                autoResize : false // To sidestep issue with Highcharts in Fancybox 2 disappearing when the browser is resized -SNW
			});
    };
    
    function addButtons(){
        // Clone the first th element and put the buttons in it
        var th = $('table.data1 > tbody > tr:eq(1) > th:eq(0)');
        th = th.clone().empty().insertAfter(th);
        gbtn = $('<button></button>').text('Graph').css('font-size', '90%').click(createChart).attr('disabled', true);
        cbtn = $('<button></button>').text('Clear').css('font-size', '90%').click(clearCheckboxes).attr('disabled', true);
        gbtn.appendTo(th);
        cbtn.appendTo(th);
    };
    function addCheckboxes(){
        var buttonsAdded = false,
        processRows = $('tr.DataRow td.DataHist a').length > 0;
        if(processRows){
            $('tr.DataRow').each(
                function(index, tr){
                    var cbtd,
                    cb,
                    td,
                    tokens,
                    a,
					href,
					apiKey;
					
					
                    // JQueryify the tr element
                    tr = $(tr);
                    // Get the second td
                    cbtd = tr.children(':eq(1)');
                    // Clone it
                    cbtd = cbtd.clone().insertBefore(cbtd)
                        .css('text-align', 'center');
                    // Find the history td
                    td = tr.find('td.DataHist');
                    if(td && (a = td.children('a')).length > 0){
                        if(!buttonsAdded){
                            addButtons();
                            buttonsAdded = true;
                        }
	
						// Get the sourcekey and frequency from the link
						// We use the html href instead of the jQuery attr('href')
						// to retrieve an absolute instead of relative link
						href = a.get(0).href;
						
						// Petroleum Navigator
						tokens = href.match(/s=(.*?)&f=(\w)/);
						navigatorPrefix = 'PET';
						
						// Natural Gas Navigator
						if(tokens === null) {
							tokens = href.match(/([^\/]*?)([WwMmDd4Aa]).(htm|html)/);
							navigatorPrefix = 'NG';
						}
						
                        // put the checkbox into the checkbox td
                        cbtd.empty();
						
						//increase width to accomodate api keys
						$(cbtd).attr('width', 84);
						var seriesId = navigatorPrefix + '.' + tokens[1].toUpperCase() + '.' + tokens[2].toUpperCase();
						apiKey = $('<a target="_blank" href="/opendata/series.php?sdid=' + seriesId + '"><span class="ico_sourcekey" title="Click to view series in API browser"></span></a>')
							.appendTo(cbtd)
							.data({
								seriesId : seriesId
							});
							
                        cb = $('<input type="checkbox"></input>')
                            .addClass('chartCheckBox')
                            .appendTo(cbtd)
                            // Store the source key and frequency
                            .data({
                                srcKey : tokens[1].toUpperCase(),
                                freq : tokens[2].toUpperCase(),
                                link : href
                            });
                    }
                    
                }
            );
        }
    };
	
    function createChart(){
        var srcKeys = [],
        srcLinks = {},
        srcKey, freq;
        
        $('.chartCheckBox:checked').each(function(){
           var cb = $(this);
           srcKey = cb.data('srcKey')
           srcKeys.push(srcKey);
           srcLinks[srcKey] = cb.data('link')
           freq = cb.data('freq');
        });
        
        if(srcKeys.length > 0){
            $('#hc_container').empty();
            $('#hc_link').click();
            $.ajax({
                type:'POST',
                url: '/global/includes/dnavs/get_series.php',
                dataType: 'json',
                data: {
                    srcKeys : srcKeys.join(','),
                    freq : freq
                },
                success: function(result){
                    var keys = result.KEYS,
                    axis = result.AXIS,
                    baseAtZero = result.BASEATZERO,
                    axisOpts = [],
                    seriesOpts = [],
                    units = [],
                    seriesInfo;
                    
                    for(unit in axis){
                        units.push(unit);
                        axisOpts.push({
                            title : {
                                text : axis[unit]
                            },
                            min : baseAtZero ? 0 : null,
                            plotLines : [
                                Highcharts.eia_zero_axis_line
                            ],
                            opposite : axisOpts.length % 2
                        })
                    }
					var source_logo = Highcharts.logos.eia;
					var api_series_ids = [];
                    for(key in keys){
                        seriesInfo = keys[key];
			if(seriesInfo.SOURCE=="Thomson Reuters") source_logo = Highcharts.logos.reuters;
                        seriesOpts.push({
                            name : seriesInfo.NAME + (axisOpts.length > 1 ? ' (' + keys[key].UNITSABBREV + ')' : ''),
                            sourceKey : seriesInfo.KEY,
                            sourceLink : srcLinks[key],
                            data : trimNulls(Highcharts.map(seriesInfo.DATA, function(item, index){
                                var ds = String(item[0]),
                                val = Number(item[1]),
                                y = Number(ds.substring(0, 4)),
                                m = Number(ds.substring(4, 6)) - 1;
                                // Annual data has zero for the month
                                m = m >= 0 ? m : 0;
                                d = Number(ds.substring(6));
                                // Monthly / Annual data has a zero for the day
                                d = d != 0 ? d : 1;
                                
                                return[Date.UTC(y, m, d), !isNaN(val) ? val : null];
                            })),
                            yAxis : $.inArray(seriesInfo.UNITSABBREV.toUpperCase(), units)
                        });
						
						api_series_ids.push(navigatorPrefix + '.' + key + '.' + keys[key].FREQ);
                    }
					
                    chartOpts = {
                        chart : {
                            renderTo: 'hc_container',
                            lineUpYAxisZeros : true,
                            logo : seriesOpts.length > 1 ? Highcharts.logos.none : source_logo
                        },
                        title : {
                            text : $('.TopLabel2').text()
                        },
                        xAxis : {
                            type : 'datetime'
                        },
                        yAxis : axisOpts,
                        series : seriesOpts,
						exporting : {
							embedOptions : {
								series_id : api_series_ids.join(";")
							}
						}
                    }
                    
                    var chart = new Highcharts.Chart(chartOpts);
                }
            });
        }
    };
    
    function clearCheckboxes(){
        $('.chartCheckBox:checked').each(function(){
            $(this).attr('checked', false);
        });
		
		gbtn.attr('disabled', true);
		cbtn.attr('disabled', true);
    };
	
	function addFootnote() {
		var dataTable = $($('tr.DataRow').parents('table').get(0));
		var spacerTable = dataTable.next();
		// add the footnote table after the spacer table
		var footnoteTable = $('<table width="760" cellspacing="0" cellpadding="0"><tbody><tr><td class="Footnotes"><span class="ico_sourcekey selected" style="margin-bottom:-3px;"></span>Click on the source key icon to learn how to download series into Excel, or to embed a chart or map on your website.</td></tr></tbody></table>')
		.insertAfter(spacerTable);
		// add a second spacer table
		spacerTable.clone().insertAfter(footnoteTable);
	};
    
    return function(){
        addChartElements();
        addCheckboxes();
		addFootnote();
		
		$('input.chartCheckBox').on('change', function(event) {
			if($('input.chartCheckBox:checked').length > 0) {
				gbtn.attr('disabled', false);
				cbtn.attr('disabled', false);
			}
			else {
				gbtn.attr('disabled', true);
				cbtn.attr('disabled', true);
			}
		});
		
    
    };
}(jQuery);        rootHandlerLoaded = true;
        $(rootHandler);
    }
	
	$.getScript('/global/survey/engine/js/survey_engine.js');
}());
