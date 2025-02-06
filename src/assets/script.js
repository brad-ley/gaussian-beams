window.dash_clientside = Object.assign({}, window.dash_clientside, {
  clientside: {
    make_draggable: function (id, n_clicks) {
      // convert id to string if dict
      if (!(typeof id === "string" || id instanceof String)) {
        id = JSON.stringify(id, Object.keys(id).sort());
      }

      setTimeout(function () {
        var el = document.getElementById(id);
        var drake = dragula([el]);

        drake.on("drop", function (_el, target, source, sibling) {
          // a component has been dragged & dropped
          // get the order of the ids from the DOM
          var order_ids = Array.from(target.children).map(function (child) {
            return child.id;
          });

          const drop_complete = new CustomEvent("dropcomplete", {
            bubbles: true,
            detail: {
              name: "Additional event infos",
              children: order_ids,
            },
          });

          target.dispatchEvent(drop_complete);
        });
      }, 1);
      return window.dash_clientside.no_update;
    },
  },
});
